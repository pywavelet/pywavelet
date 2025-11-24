import glob
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy.signal import chirp
from scipy.special import betainc
from tqdm.auto import trange

jax.config.update("jax_enable_x64", True)

DEVICE = jax.devices()[0].device_kind
print(f"Running JAX on {DEVICE}")

PI = np.pi
FREQ_RANGE = [20, 100]


def omega(Nf: int, Nt: int):
    df = 2 * np.pi / (Nf * Nt)
    return df * np.arange(-Nt // 2, Nt // 2, dtype=np.float64)


def phitilde_vec_norm(Nf: int, Nt: int, d: float):
    return _phitilde_vec(omega(Nf, Nt), Nf, d) * np.sqrt(np.pi)


def phi_vec(Nf: int, d: float = 4.0, q: int = 16):
    """get time domain phi as fourier transform of _phitilde_vec
    q: number of Nf bins over which the window extends?
    """
    insDOM = 1.0 / np.sqrt(np.pi / Nf)
    K = q * 2 * Nf
    half_K = q * Nf  # xp.int64(K/2)

    dom = 2 * np.pi / K  # max frequency is K/2*dom = pi/dt = OM
    DX = np.zeros(K, dtype=np.complex128)

    # zero frequency
    DX[0] = insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1 : half_K + 1] = _phitilde_vec(dom * np.arange(1, half_K + 1), Nf, d)
    # negative frequencies
    DX[half_K + 1 :] = _phitilde_vec(
        -dom * np.arange(half_K - 1, 0, -1), Nf, d
    )
    DX = K * np.fft.ifft(DX, K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(2.0) / np.sqrt(K / dom)  # *xp.linalg.norm(phi)

    phi *= nrm
    return np.array(phi)


def _phitilde_vec(omega, Nf: int, d: float = 4.0):
    """Compute phi_tilde(omega_i) array, nx is filter steepness, defaults to 4.
    Eq 11 of Cornish '20
    """
    dF = 1.0 / (2 * Nf)  # NOTE: missing 1/dt?
    dOmega = 2 * np.pi * dF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_dOmega = 1.0 / np.sqrt(dOmega)

    A = dOmega / 4
    B = dOmega - 2 * A  # Cannot have B \leq 0.
    if B <= 0:
        raise ValueError("B must be greater than 0")

    phi = np.zeros(omega.size, dtype=np.float64)
    mask = (A <= np.abs(omega)) & (np.abs(omega) < A + B)
    vd = (np.pi / 2.0) * betainc(d, d, (np.abs(omega)[mask] - A) / B)
    phi[mask] = inverse_sqrt_dOmega * np.cos(vd)
    phi[np.abs(omega) < A] = inverse_sqrt_dOmega
    return phi


def Cnm_matrix(nt: int, nf: int) -> jnp.ndarray:
    """
    C[i,j] == 1   if (i + j) % 2 == 0
           == 1j  otherwise
    """
    i = jnp.arange(nt)[:, None]  # shape (nt,1)
    j = jnp.arange(nf)[None, :]  # shape (1,nf)
    return jnp.where((i + j) % 2 == 0, 1.0, 1j).astype(jnp.complex128)


def make_gif_from_images(image_dir, gif_name, duration=3):
    images = []
    files = glob.glob(os.path.join(image_dir, "*.png"))
    files = sorted(
        files,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    for filename in files:
        images.append(Image.open(filename))
    images[0].save(
        gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration * 1000,  # Convert seconds to milliseconds
        loop=0,
    )


def make_animation():
    Nf = 32
    Nt = Nf
    nx = 4.0

    assert Nf & (Nf - 1) == 0, "Nf must be a power of 2"
    fs = 256
    dt = 1 / fs
    mult = 16
    nx = 1
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    f = np.fft.rfftfreq(ND, dt)
    T = ND * dt
    y = chirp(
        t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic"
    )
    phif = jnp.array(phitilde_vec_norm(Nf, Nt, d=nx))
    phit = phi_vec(Nf, d=nx)
    yf = jnp.fft.fft(y)

    # wave = transform_wavelet_freq_helper_numba(yf, Nf, Nt, phif)
    yf = jnp.array(yf)
    jax_phif = jnp.array(phif)

    plt.plot(phitilde_vec_norm(Nf, Nt, d=1), label="d=1")
    plt.plot(phitilde_vec_norm(Nf, Nt, d=4), label="d=4")
    plt.plot(phitilde_vec_norm(Nf, Nt, d=100), label="d=100")
    plt.legend()
    plt.show()

    wave = freq_to_wdm(yf, Nf, Nt, jax_phif)
    delta_T = T / Nt
    delta_F = 1 / (2 * delta_T)
    t_bins = np.arange(0, Nt) * delta_T
    f_bins = np.arange(0, Nf) * delta_F

    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    axes[0].semilogy(f, (np.abs(yf) ** 2)[: ND // 2 + 1], label="FFT")
    axes[0].set_xlabel("Frequency (Hz)", fontsize=14)
    axes[0].set_ylabel("Power", fontsize=14)
    axes[0].set_xlim(10, 100)
    axes[0].set_ylim(bottom=10)

    im = axes[1].imshow(
        np.abs((wave.T)),
        aspect="auto",
        extent=[t_bins[0], t_bins[-1], f_bins[0], f_bins[-1]],
        origin="lower",
        interpolation="nearest",
    )
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=14)
    plt.tight_layout()

    fig.subplots_adjust(hspace=0.5)
    plt.show()

    plt_dir = "wdm_plots"
    # make aimation
    os.makedirs(plt_dir, exist_ok=True)

    f = np.fft.fftfreq(ND, dt)
    f = np.fft.fftshift(f)
    d = jnp.fft.fftshift(yf)
    output = jnp.zeros((Nf, Nt), dtype=jnp.complex128)
    i0 = (Nt * Nf) // 2 + 1
    j0 = i0 - (Nt // 2)
    j1 = i0 + (Nt // 2)
    min_wnm, max_wnm = np.min(wave), np.max(wave)

    for m in range(Nt):
        rolled_d = jnp.roll(d, -(m * Nt // 2))
        xni = jnp.fft.ifft(rolled_d[j0:j1] * phif)

        fig = plt.figure(figsize=(5, 6))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        ax1.plot(f, np.abs(d) ** 2, color="tab:gray", alpha=0.2)
        ax1.plot(f, np.abs(rolled_d) ** 2, color="tab:gray", label="Roll(y)")
        ax1.fill_between(
            np.arange(-Nt // 2, Nt / 2),
            max(np.abs(d) ** 2) * phif / max(phif),
            alpha=0.5,
            color="tab:orange",
            label="Window",
        )
        ax1.legend(loc="upper left", frameon=False)
        ax1.set_ylim(bottom=0)
        ax2.plot(t_bins, xni)
        ax2.set_ylim(min_wnm, max_wnm)

        ax1.set_xlabel("Freq [Hz]")
        ax1.set_ylabel("Power")

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Wnm")

        wdm_m = np.array(wave)
        wdm_m[:, m + 1 :] = 0
        # cmap of 'bwr' with vcenter=0

        curr_min = min(wdm_m.min(), -1)
        curr_max = max(wdm_m.max(), 1)

        norm = TwoSlopeNorm(0, curr_min, curr_max)
        im = ax3.imshow(
            wdm_m.T,
            aspect="auto",
            extent=[t_bins[0], t_bins[-1], f_bins[0], f_bins[-1]],
            origin="lower",
            interpolation="nearest",
            norm=norm,
            cmap="bwr",
        )
        ax3.set_xlabel("Time [s]", fontsize=14)
        ax3.set_ylabel("Freq [Hz]", fontsize=14)
        # add colorbar and label it Wnm
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label("Wnm", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{plt_dir}/wdm_{m:02d}.png")

    make_gif_from_images(plt_dir, "wdm.gif", duration=0.5)


def gnm_matrix(Nf, Nt, T, d=4.0):
    """
    g˜nm(ω) = e −inω∆T (CnmΦ(ω − m∆Ω)+C∗nmΦ(ω + m∆Ω))
    """
    cnm = Cnm_matrix(Nf, Nt)
    cnmconj = jnp.conjugate(cnm)
    omegas = omega(Nf, Nt)
    delta_T = T / Nt
    delta_F = 1 / (2 * delta_T)
    delta_omega = 2 * PI * delta_F
    # gnm (mesh of Nt, Nf, omega)
    nomega = len(omegas)
    gnm_t = np.zeros((Nf, Nt, nomega), dtype=complex)
    for n in trange(Nf):
        for m in range(Nt):
            exp = np.exp(-1j * n * omegas * delta_T)
            phi1 = _phitilde_vec(omegas - m * delta_omega, Nf, d)
            phi2 = _phitilde_vec(omegas + m * delta_omega, Nf, d)
            gnm_t[n, m, :] = exp * (cnm[n, m] * phi1 + cnmconj[n, m] * phi2)
    gnm_t = np.array(gnm_t)
    gnm = np.fft.ifft(gnm_t, axis=-1)
    return gnm


gnm = gnm_matrix(Nf=32, Nt=64, T=4.0, d=4.0)


def plot_gnm(t=0, f=0):
    plt.close("all")
    plt.plot(gnm[t, f, :].real)
    plt.show()


def phi_unit(f: np.ndarray, A: float, d: float) -> np.ndarray:
    B = 1.0 - 2.0 * A
    absf = np.abs(f)

    phi = np.zeros_like(f, dtype=np.float64)

    # region |f| < A
    mask0 = absf < A
    phi[mask0] = 1.0

    # region A <= |f| < A + B
    mask1 = (absf >= A) & (absf < A + B)
    x = (absf[mask1] - A) / B
    p = betainc(d, d, x)
    phi[mask1] = np.cos(np.pi * p / 2.0)

    # region |f| >= A + B stays zero
    return phi


def gnm_matrix(
    nt: int,
    nf: int,
    dt: float,
    d: float = 4.0,
) -> np.ndarray:
    # total time samples
    nd = nt * nf
    # time and frequency steps
    dT = nf * dt
    dF = 1.0 / (2.0 * dT)
    dOmega = 2 * np.pi * dF  # Near Eq 10 # 2 pi times DF
    A = dOmega / 4

    # fetch your C matrix; must return shape (nt, nf)
    C = Cnm_matrix(nt, nf)

    # build grids for broadcasting
    ns = np.arange(nt).reshape(nt, 1, 1)  # (nt,1,1)
    ms = np.arange(nf).reshape(1, nf, 1)  # (1,nf,1)
    fs = np.fft.fftfreq(nd, d=dt).reshape(1, 1, nd)  # (1,1,n)
    C = C.reshape(nt, nf, 1)  # (nt,nf,1)

    # compute the two phi_unit terms
    Phi1 = phi_unit(fs / dF - ms, A, d)
    Phi2 = phi_unit(fs / dF + ms, A, d)

    # the complex exponential prefactor
    exp_fac = np.exp(-1j * ns * 2.0 * np.pi * fs * dT)

    # assemble
    out = exp_fac * (C * Phi1 + np.conj(C) * Phi2) / np.sqrt(2.0 * dF)
    return np.fft.ifft(out, axis=2)


Nf = 32
Nt = Nf
nx = 4.0

assert Nf & (Nf - 1) == 0, "Nf must be a power of 2"
fs = 256
dt = 1 / fs
mult = 16
nx = 4
ND = Nt * Nf
t = np.arange(0, ND) * dt
f = np.fft.rfftfreq(ND, dt)
T = ND * dt
y = chirp(t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic")
phif = jnp.array(phitilde_vec_norm(Nf, Nt, d=nx))
phit = phi_vec(Nf, d=nx)
yf = jnp.fft.fft(y)
wnm = freq_to_wdm(yf, Nf, Nt, phif)
gnm = gnm_matrix(nt=Nt, nf=Nf, dt=dt, d=4.0)


# #recon_x[k] = Sum Sum gnm[k] * wnm
# recon_x = np.zeros(ND)
# for k in range(ND):
#     for m in range(Nt):
#         for n in range(Nf):
#             # plt.plot(gnm[m, n, k], label='gnm')
#             # plt.plot(wnm[n, m], label='wnm')
#             # plt.
#             # plt.legend()
#
#             recon_x[k] += float((gnm[m, n, k] * wnm[n, m]).real)
#
#


# def x_i(m: int, data: jnp.ndarray, Nf: int, Nt: int, phit: jnp.ndarray) -> jnp.ndarray:
#     # i0 = data.shape[0] // 2 + 1
#     # i0 is the center of the data -- used
#     i0 = (Nt * Nf) // 2 + 1
#     j0 = i0 - (Nt // 2)
#     j1 = i0 + (Nt // 2)
#     xni = jnp.fft.ifft(jnp.roll(data, - (m * Nt // 2))[j0: j1] * phif)
#     return xni


def xtilde_i(
    m: int, data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray
) -> jnp.ndarray:
    # i0 = data.shape[0] // 2 + 1
    # i0 is the center of the data -- used
    i0 = (Nt * Nf) // 2 + 1
    j0 = i0 - (Nt // 2)
    j1 = i0 + (Nt // 2)
    xni = jnp.fft.ifft(jnp.roll(data, -(m * Nt // 2))[j0:j1] * phif)
    return xni


def freq_to_wdm(
    data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns Nt, Nf
    """
    # fftshifft data to match up with Neil Eq 17
    data_shifted = jnp.fft.fftshift(data)
    output = jnp.zeros((Nf, Nt), dtype=jnp.complex128)
    # i is shift in freq
    return (
        jax.lax.fori_loop(
            0,
            Nt,
            lambda i, output: output.at[:, i].set(
                xtilde_i(i, data_shifted, Nf, Nt, phif)
            ),
            output,
        )
        * Cnm_matrix(Nf, Nt)
    ).real * jnp.sqrt(2.0)


def wdm_to_freq(wnm: jnp.ndarray, phit: jnp.ndarray) -> jnp.ndarray:
    Nt, Nf = wnm.shape
    output = np.zeros(Nf * Nt, dtype=np.complex128)
    # 1) FFT each row along time‐axis
    # 2) multiply each row by the filter
    # 3) compute roll‐offsets for each row
    # 4) build an index array to do all the rolls in one go:
    #    for each m, pick Wf[m, (j - shifts[m]) % Nt]
    # 5) return as a flat length‐Nf*Nt vector
    W = jnp.fft.fft(wnm, axis=0)  # shape (Nf, Nt), complex
    Wf = jnp.sum(W * phit[:, None], axis=0)  # phit --> should be Nt long
    # Wf --> Nf size
    shifts = (jnp.arange(Nf) * (Nt // 2)) % Nt  # shape (Nf,)

    j = jnp.arange(Nt)  # (Nt,)
    idx = (j[None, :] - shifts[:, None]) % Nt  # (Nf, Nt)

    rolled = jnp.take_along_axis(Wf, idx, axis=1)  # (Nf, Nt)

    return rolled.reshape(-1)


def wdm_to_freq(
    wave: np.ndarray,
    phi: np.ndarray,
    dt: float,
) -> np.ndarray:
    Nt, Nf = wave.shape
    ND = Nt * Nf

    M = Nf
    omega = np.pi / dt
    delta_omega = omega / M
    B = omega / (2 * M)
    A = (delta_omega - B) / 2.0
    scale = np.sqrt(np.pi) / dt

    dataf = np.zeros(ND, dtype=float)

    halfNt = Nt // 2

    for i in range(1, Nf - 1):
        # build complex row of length Nt
        real = np.zeros(Nt, dtype=float)
        imag = np.zeros(Nt, dtype=float)

        # fill real/imag parts according to (i+j)%2 rule
        sign_flip = -1 if (i % 2 == 0) else 1
        for j in range(Nt):
            if (i + j) % 2 == 0:
                real[j] = wave[j, i]
            else:
                imag[j] = sign_flip * wave[j, i]

        row = real + 1j * imag
        row_fft = np.fft.fft(row)

        jj = i * halfNt
        y = scale if (i % 2 == 0) else -scale

        # Positive frequencies (j = 0 .. halfNt-1)
        for j in range(halfNt):
            x = y * phi[j]
            kk = jj + j
            dataf[kk] += x * row_fft[j].real
            dataf[ND - kk] += x * row_fft[j].imag

        # “Negative” freqs (j = halfNt-1 .. 1)
        for j in range(halfNt - 1, 0, -1):
            x = y * phi[j]
            kk = jj - j
            # map to row_fft[Nt-j]
            dataf[kk] += x * row_fft[Nt - j].real
            dataf[ND - kk] += x * row_fft[Nt - j].imag

    return dataf


#


new_x = wdm_to_freq(wnm, phif, dt)
plt.close("all")
# plt subplots
fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
axes[0].semilogy(f, (np.abs(yf) ** 2)[: ND // 2 + 1], label="FFT")
axes[0].set_xlabel("Frequency (Hz)", fontsize=14)
axes[1].semilogy(f, (np.abs(new_x) ** 2)[: ND // 2 + 1], label="FFT")
axes[0].set_ylabel("Power", fontsize=14)

plt.show()
