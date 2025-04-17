from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from numba import njit
from scipy.special import betainc

jax.config.update("jax_enable_x64", True)


def inverse_wavelet_freq_helper_np(
    wave_in: np.ndarray, phif: np.ndarray, Nf: int, Nt: int
) -> np.ndarray:
    """jit compatible loop for inverse_wavelet_freq"""
    wave_in = wave_in.T
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt, np.complex128)
    res = np.zeros(ND // 2 + 1, dtype=np.complex128)
    __core(Nf, Nt, prefactor2s, wave_in, phif, res)

    return res


def __core(
    Nf: int,
    Nt: int,
    prefactor2s: np.ndarray,
    wave_in: np.ndarray,
    phif: np.ndarray,
    res: np.ndarray,
) -> None:
    for m in range(0, Nf + 1):
        __pack_wave_inverse(m, Nt, Nf, prefactor2s, wave_in)
        fft_prefactor2s = np.fft.fft(prefactor2s)
        __unpack_wave_inverse(m, Nt, Nf, phif, fft_prefactor2s, res)


@njit()
def __pack_wave_inverse(
    m: int, Nt: int, Nf: int, prefactor2s: np.ndarray, wave_in: np.ndarray
) -> None:
    """helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m == 0:
        for n in range(0, Nt):
            prefactor2s[n] = 2 ** (-1 / 2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(0, Nt):
            prefactor2s[n] = 2 ** (-1 / 2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(0, Nt):
            val = wave_in[n, m]
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2 * val


@njit()
def __unpack_wave_inverse(
    m: int,
    Nt: int,
    Nf: int,
    phif: np.ndarray,
    fft_prefactor2s: np.ndarray,
    res: np.ndarray,
) -> None:
    """helper for unpacking results of frequency domain inverse transform"""

    if m == 0 or m == Nf:
        for i_ind in range(0, Nt // 2):
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = (2 * i) % Nt
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == Nf:
            i_ind = Nt // 2
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (Nt // 2 * m) % Nt
        ind32 = (Nt // 2 * m) % Nt
        for i_ind in range(0, Nt // 2):
            i1 = Nt // 2 * m - i_ind
            i2 = Nt // 2 * m + i_ind
            # assert ind31 == i1%Nt
            # assert ind32 == i2%Nt
            res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31 < 0:
                ind31 = Nt - 1
            if ind32 == Nt:
                ind32 = 0
        res[Nt // 2 * m] = fft_prefactor2s[(Nt // 2 * m) % Nt] * phif[0]


@partial(jit, static_argnames=("Nf", "Nt"))
def inverse_wavelet_freq_helper_jax(
    wave_in: jnp.ndarray, phif: jnp.ndarray, Nf: int, Nt: int
) -> jnp.ndarray:
    """JAX vectorized function for inverse_wavelet_freq with corrected shapes and ranges."""
    wave_in = wave_in.T
    ND = Nf * Nt

    m_range = jnp.arange(Nf + 1)
    prefactor2s = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    n_range = jnp.arange(Nt)

    # Handle m=0 and m=Nf cases
    prefactor2s = prefactor2s.at[0].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0]
    )
    prefactor2s = prefactor2s.at[Nf].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0]
    )

    # Handle middle m cases
    m_mid = m_range[1:Nf]
    n_grid, m_grid = jnp.meshgrid(n_range, m_mid, indexing="ij")
    val = wave_in[n_grid, m_grid]
    mult2 = jnp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s = prefactor2s.at[1:Nf].set((mult2 * val).T)

    fft_prefactor2s = jnp.fft.fft(prefactor2s, axis=1)

    res = jnp.zeros(ND // 2 + 1, dtype=jnp.complex128)

    # Unpack for m=0 and m=Nf
    i_ind_range = jnp.arange(Nt // 2)
    i_0 = i_ind_range
    ind3_0 = (2 * i_0) % Nt
    res = res.at[i_0].add(fft_prefactor2s[0, ind3_0] * phif[i_ind_range])

    i_Nf = jnp.abs(Nf * (Nt // 2) - i_ind_range)
    ind3_Nf = (2 * i_Nf) % Nt
    res = res.at[i_Nf].add(fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range])

    special_index = jnp.abs(Nf * (Nt // 2) - (Nt // 2))
    res = res.at[special_index].add(fft_prefactor2s[Nf, 0] * phif[Nt // 2])

    # Unpack for middle m values
    m_mid = m_range[1:Nf]
    i_ind_range_mid = jnp.arange(Nt // 2)
    m_grid_mid, i_ind_grid_mid = jnp.meshgrid(
        m_mid, i_ind_range_mid, indexing="ij"
    )
    i1 = (Nt // 2) * m_grid_mid - i_ind_grid_mid
    i2 = (Nt // 2) * m_grid_mid + i_ind_grid_mid
    ind31 = i1 % Nt
    ind32 = i2 % Nt

    res = res.at[i1].add(
        fft_prefactor2s[m_grid_mid, ind31] * phif[i_ind_grid_mid]
    )
    res = res.at[i2].add(
        fft_prefactor2s[m_grid_mid, ind32] * phif[i_ind_grid_mid]
    )

    # Correct the center points for middle m's
    center_indices = (Nt // 2) * m_mid
    fft_indices = center_indices % Nt
    values = fft_prefactor2s[m_mid, fft_indices] * phif[0]
    res = res.at[center_indices].set(values)

    return res


def phitilde_vec_norm(Nf: int, Nt: int, d: float) -> np.ndarray:
    """Normalize phitilde for inverse frequency domain transform."""
    df = 2 * np.pi / (Nf * Nt)
    omega = df * np.arange(0, Nt // 2 + 1, dtype=np.float64)

    dF = 1.0 / (2 * Nf)
    dOmega = 2 * np.pi * dF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_dOmega = 1.0 / np.sqrt(dOmega)

    A = dOmega / 4
    B = dOmega - 2 * A  # Cannot have B \leq 0.
    assert B > 0, "B must be greater than 0"

    phi = np.zeros(omega.size, dtype=np.float64)
    mask = (A <= np.abs(omega)) & (np.abs(omega) < A + B)  # Minor changes
    vd = (np.pi / 2.0) * betainc(d, d, (np.abs(omega[mask]) - A) / B)
    phi[mask] = inverse_sqrt_dOmega * np.cos(vd)
    phi[np.abs(omega) < A] = inverse_sqrt_dOmega
    return np.array(phi) * np.sqrt(np.pi)


A = 2.0
dt = 1 / 32
f0 = 8.0
Nt, Nf = 2**3, 2**3
ND = Nt * Nf

wnm = np.zeros((Nt, Nf))
m0 = int(f0 * ND * dt)
f0_bin_idx = int(2 * m0 / Nt)
odd_t_indices = np.arange(Nt) % 2 != 0
wnm[odd_t_indices, f0_bin_idx] = A * np.sqrt(2 * Nf)
T = Nt * Nf * dt
delta_T = T / Nt
delta_F = 1 / (2 * delta_T)
tbins = np.arange(0, Nt) * delta_T
fbins = np.arange(0, Nf) * delta_F

numerical_wnm = np.array(
    [
        [
            -2.17231658e-16,
            -2.17231658e-16,
            1.82849722e-15,
            1.82849722e-15,
            -1.59693281e-15,
            -1.59693281e-15,
            -1.14095073e-15,
            -1.14095073e-15,
        ],
        [
            5.21849164e-15,
            -1.92598987e-15,
            1.69405022e-15,
            -6.11340860e-18,
            4.95368161e-16,
            6.95072219e-16,
            1.56088961e-14,
            -3.23455137e-15,
        ],
        [
            6.27820524e-15,
            -7.39243293e-16,
            -1.75734889e-15,
            -1.08780986e-14,
            3.19246930e-15,
            -1.41771970e-14,
            1.35153646e-15,
            -1.52447448e-14,
        ],
        [
            -1.35115223e-14,
            1.52017787e-15,
            4.31116053e-15,
            -3.99699681e-16,
            -8.63776618e-16,
            5.22364583e-16,
            -1.34704878e-14,
            -1.09012981e-15,
        ],
        [
            -1.92753121e-14,
            7.99999952e00,
            -6.09730187e-15,
            7.99999952e00,
            -1.89903889e-14,
            7.99999952e00,
            -2.70376169e-14,
            7.99999952e00,
        ],
        [
            1.35115223e-14,
            1.52017787e-15,
            -4.31116053e-15,
            -3.99699681e-16,
            8.63776618e-16,
            5.22364583e-16,
            1.34704878e-14,
            -1.09012981e-15,
        ],
        [
            6.27820524e-15,
            7.39243293e-16,
            -1.75734889e-15,
            1.08780986e-14,
            3.19246930e-15,
            1.41771970e-14,
            1.35153646e-15,
            1.52447448e-14,
        ],
        [
            -5.21849164e-15,
            -1.92598987e-15,
            -1.69405022e-15,
            -6.11340860e-18,
            -4.95368161e-16,
            6.95072219e-16,
            -1.56088961e-14,
            -3.23455137e-15,
        ],
    ]
)

t = np.arange(0, ND) * dt
time_data = A * np.sin(2 * np.pi * f0 * t)
freq_data = np.fft.rfft(time_data)
freqs = np.fft.rfftfreq(ND, d=dt)


phif = phitilde_vec_norm(Nf, Nt, 4.0)


freq_np = inverse_wavelet_freq_helper_np(
    numerical_wnm, phif, Nf, Nt
) / np.sqrt(2)
freq_jax = inverse_wavelet_freq_helper_jax(numerical_wnm, phif, Nf, Nt)
freq_jax = freq_jax / np.sqrt(2)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pcolor(tbins, fbins, wnm.T, shading="auto", cmap="viridis")

ax[1].loglog(
    freqs,
    np.abs(freq_data) ** 2,
    label="Original Signal",
    alpha=0.5,
    color="black",
    lw=1,
)
ax[1].plot(
    freqs,
    np.abs(freq_np) ** 2,
    label="NumPy Inverse",
    alpha=0.5,
    lw=2,
    marker="o",
)
ax[1].plot(
    freqs,
    np.abs(freq_jax) ** 2,
    label="JAX Inverse",
    alpha=0.5,
    lw=2,
    marker="s",
)
ax[1].legend()
plt.ylabel("Amplitude")
plt.title("Inverse Wavelet Transform Comparison")
plt.show()
