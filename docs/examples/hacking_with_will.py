import numpy as np
from numpy import fft
from scipy.special import betainc
from scipy.signal import chirp
from typing import List, Tuple
import jax

jax.config.update('jax_enable_x64', True)

DEVICE = jax.devices()[0].device_kind
print(f"Running JAX on {DEVICE}")

PI = np.pi
FREQ_RANGE = [20, 100]


def omega(Nf: int, Nt: int):
    """Get the angular frequencies of the time domain signal."""
    df = 2 * np.pi / (Nf * Nt)
    return df * np.arange(-Nt // 2, Nt // 2, dtype=np.float64)


def phitilde_vec_norm(
        Nf: int, Nt: int, d: float
):
    """Normalize phitilde for inverse frequency domain transform."""
    omegas = omega(Nf, Nt)
    _phi_t = _phitilde_vec(omegas, Nf, d) * np.sqrt(np.pi)
    return np.array(_phi_t)


def phi_vec(
        Nf: int, d: float = 4.0, q: int = 16
):
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
    DX[1: half_K + 1] = _phitilde_vec(dom * np.arange(1, half_K + 1), Nf, d)
    # negative frequencies
    DX[half_K + 1:] = _phitilde_vec(
        -dom * np.arange(half_K - 1, 0, -1), Nf, d
    )
    DX = K * np.fft.ifft(DX, K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(2.0) / np.sqrt(K / dom)  # *xp.linalg.norm(phi)

    phi *= nrm
    return np.array(phi)


def _phitilde_vec(
        omega, Nf: int, d: float = 4.0
):
    """Compute phi_tilde(omega_i) array, nx is filter steepness, defaults to 4.

    Eq 11 of https://arxiv.org/pdf/2009.00043.pdf (Cornish et al. 2020)

    phi(omega_i) =
        1/sqrt(2π∆F) if |omega_i| < A
        1/sqrt(2π∆F) cos(nu_d π/2 * |omega|-A / B) if A < |omega_i| < A + B

    Where nu_d = normalized incomplete beta function

    Parameters
    ----------
    ω : xp.ndarray
        Array of angular frequencies
    Nf : int
        Number of frequency bins
    d : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.

    Returns
    -------
    xp.ndarray
        Array of phi_tilde(omega_i) values

    """
    dF = 1.0 / (2 * Nf)  # NOTE: missing 1/dt?
    dOmega = 2 * np.pi * dF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_dOmega = 1.0 / np.sqrt(dOmega)

    A = dOmega / 4
    B = dOmega - 2 * A  # Cannot have B \leq 0.
    if B <= 0:
        raise ValueError("B must be greater than 0")

    phi = np.zeros(omega.size, dtype=np.float64)
    mask = (A <= np.abs(omega)) & (np.abs(omega) < A + B)  # Minor changes
    vd = (np.pi / 2.0) * _nu_d(omega[mask], A, B, d=d)  # different from paper
    phi[mask] = inverse_sqrt_dOmega * np.cos(vd)
    phi[np.abs(omega) < A] = inverse_sqrt_dOmega
    return phi


def _nu_d(
        omega, A: float, B: float, d: float = 4.0
):
    """Compute the normalized incomplete beta function.

    Parameters
    ----------
    omega : np.ndarray
        Array of angular frequencies
    A : float
        Lower bound for the beta function
    B : float
        Upper bound for the beta function
    d : float, optional
        Controlls the 'steepness'' -- 100 --> sqr wave, lower --> smoother roll


    Returns
    -------
    np.ndarray
        Array of ν_d values

    scipy.special.betainc
    https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.special.betainc.html

    """
    x = (np.abs(omega) - A) / B
    return betainc(d, d, x)


def simulate_data(Nf):
    # assert Nf is power of 2
    assert Nf & (Nf - 1) == 0, "Nf must be a power of 2"
    fs = 512
    dt = 1 / fs
    Nt = Nf
    mult = 16
    nx = 4.0
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    y = chirp(t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic")
    phif = phitilde_vec_norm(Nf, Nt, d=nx)
    yf = fft.fft(y)[:ND // 2 + 1]

    return yf, phif


import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

#
# # collect plotting data
# Nf = Nt = 2 ** 6
# yf, phif = simulate_data(Nf)
# wave = transform_wavelet_freq_helper_numba(yf, Nf, Nt, phif)
# jax_yf = jnp.array(yf)
# jax_phif = jnp.array(phif)
# wave_jax = transform_wavelet_freq_helper_JAX(jax_yf, Nf, Nt, jax_phif)
# nplots = 3
#
# print(f"Phif shape {phif.shape}")
# print(f"data shape {jax_yf.shape}")
#
#
# def xn_i(m: int, data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray) -> jnp.ndarray:
#     xni = jnp.fft.ifft(jnp.roll(data, - (m * Nt // 2))[:(Nt // 2) + 1] * phif)
#     # print(xni.shape)
#     return xni
#
#
# def transform_conv(data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray) -> jnp.ndarray:
#     # fftshifft data to match up with Neil Eq 17
#     data_shifted = jnp.fft.fftshift(data)
#
#     xn = jnp.array([
#         xn_i(i, data_shifted, Nf, Nt, phif)  for i in range(Nf + 1)
#     ])
#
#
#
#     #
#     #
#     # wave = jax.lax.fori_loop(
#     #     0, Nt,
#     #     lambda i, accum: jnp.stack((accum, xn_i(i, data, Nf, Nt, phif)), 1),
#     #     jnp.zeros((Nt, 1), dtype=jnp.complex64)
#     # )
#     return wave[:, 1:]
#
#
# wave_conv = transform_conv(jax_yf, Nf, Nt, jax_phif)
#
# # render plot
# fig, ax = plt.subplots(1, nplots, figsize=(5, 5), sharex=True, sharey=True)
# ax[0].imshow(np.abs(np.rot90(wave)))
# ax[0].set_title("Numba")
# ax[1].imshow(np.abs(np.rot90(wave_jax)))
# ax[1].set_title("Jax")
# ax[2].imshow(np.abs(np.rot90(wave_conv)))
# ax[2].set_title("JaxCONV")
#
# for a in ax:
#     a.set_xticks([])
#     a.set_yticks([])
#     a.set_ylim(120, 70)
# plt.tight_layout()
# plt.show()

Nf = 32
Nt = Nf
nx = 4.0


def simulate_data(Nf):
    # assert Nf is power of 2
    assert Nf & (Nf - 1) == 0, "Nf must be a power of 2"
    fs = 512
    dt = 1 / fs
    Nt = Nf
    mult = 16
    nx = 1
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    y = chirp(t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic")
    phif = phitilde_vec_norm(Nf, Nt, d=nx)
    phit = phi_vec(Nf, d=nx)
    yf = fft.fft(y)
    return y, yf, phif, phit


y, yf, phif, phit = simulate_data(Nf)
# wave = transform_wavelet_freq_helper_numba(yf, Nf, Nt, phif)
jax_yf = jnp.array(yf)
jax_phif = jnp.array(phif)

plt.plot(phitilde_vec_norm(Nf, Nt, d=1), label='d=1')
plt.plot(phitilde_vec_norm(Nf, Nt, d=4), label='d=4')
plt.plot(phitilde_vec_norm(Nf, Nt, d=100), label='d=100')
plt.legend()
plt.show()


def xtilden_i(m: int, data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray) -> jnp.ndarray:
    i0 = data.shape[0] // 2 + 1
    Ntby2 = Nt // 2
    # i0 is the center of the data -- used
    xni = jnp.fft.ifft(jnp.roll(data, - (m * Ntby2))[i0 - (Ntby2):i0 + (Ntby2)] * phif)
    return xni


def Cnm_matrix(Nf: int, Nt: int) -> jnp.ndarray:
    i_inds = jnp.arange(Nf)
    j_inds = jnp.arange(Nt)

    ij = i_inds[:, None] + j_inds[None, :]
    ij_prod = i_inds[:, None] * j_inds[None, :]

    sign_matrix = jnp.where(ij_prod % 2 == 0, 1, -1)

    return jnp.where(ij % 2 == 0, 1, 1j) * sign_matrix


def freq_to_wdm(data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray) -> jnp.ndarray:
    # fftshifft data to match up with Neil Eq 17
    data_shifted = jnp.fft.fftshift(data)
    output = jnp.zeros((Nf, Nt), dtype=jnp.complex128)
    return (jax.lax.fori_loop(
        0, Nt,
        lambda i, output: output.at[:, i].set(xtilden_i(i, data_shifted, Nf, Nt, phif)),
        output
    ) * Cnm_matrix(Nf, Nt)).real * jnp.sqrt(2.0)




#
#
# def gnm_matrix(omega, Nf, Nt, A, B, d):
#     """
#     g˜nm(ω) = e −inω∆T (CnmΦ(ω − m∆Ω)+C∗nmΦ(ω + m∆Ω))
#
#
#     2A+B = ∆Ω
#     """
#
#
#     cnm = Cnm_matrix(Nf, Nt)
#     cnmconj = jnp.conjugate(cnm)
#
#
#
#     v1 = jnp.exp(-1j * omega[:, None] * jnp.arange(Nt) / Nf)
#
#     phi =
#
#
#     return  v1 * cnm * ph +

#
# def gnm(Cnm, CnmConj, A, B, d):  ### Eq 10
#     """
#     gnm(ω) = e −inω∆T (CnmΦ(ω − m∆Ω)+C∗nmΦ(ω + m∆Ω))
#     2A+B = ∆Ω
#     """
#
#     # omega = jnp.fft.fftshift(omega)
#     # omega = jnp.roll(omega, -Nf // 2)
#     # omega = jnp.fft.ifftshift(omega)
#
#      _phitilde_vec(omegas, Nf, d) * np.sqrt(np.pi)
#
#     return v1 * Cnm * ph + v1 * CnmConj * ph


#
# def

wave = freq_to_wdm(jax_yf, Nf, Nt, jax_phif)
# wave2 = freq_to_wdm(jax_yf, Nf, Nt, jax_phif)

T = 4.0
delta_T = T / Nt
delta_F = 1 / (2 * delta_T)
t_bins = np.arange(0, Nt) * delta_T
f_bins = np.arange(0, Nf) * delta_F


plt.imshow(np.abs(np.rot90(wave)))
plt.show()
