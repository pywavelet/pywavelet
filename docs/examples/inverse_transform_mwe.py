from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import fft

import numpy as np
from numba import njit
from numpy import fft
from scipy.special import betainc


def inverse_wavelet_freq_helper_fast(
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
def inverse_wavelet_freq_helper(
        wave_in: jnp.ndarray, phif: jnp.ndarray, Nf: int, Nt: int
) -> jnp.ndarray:
    """JAX vectorized function for inverse_wavelet_freq with corrected shapes and ranges."""
    # Transpose to match the NumPy version.
    wave_in = wave_in.T
    ND = Nf * Nt

    # Allocate prefactor2s for each m (shape: (Nf+1, Nt)).
    m_range = jnp.arange(Nf + 1)
    prefactor2s = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    n_range = jnp.arange(Nt)

    # m == 0 case
    prefactor2s = prefactor2s.at[0].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0]
    )

    # m == Nf case
    prefactor2s = prefactor2s.at[Nf].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0]
    )

    # Other m cases: use meshgrid for vectorization.
    m_mid = m_range[1:Nf]
    # Create grids: n_grid (columns) and m_grid (rows)
    n_grid, m_grid = jnp.meshgrid(n_range, m_mid)
    # Index the transposed wave_in using (n, m) as in the NumPy version.
    val = wave_in[n_grid, m_grid]
    # Apply the alternating multiplier based on (n+m) parity.
    mult2 = jnp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s = prefactor2s.at[1:Nf].set(mult2 * val)

    # Apply FFT along axis 1 for all m.
    fft_prefactor2s = fft(prefactor2s, axis=1)

    # Allocate the result array with corrected shape.
    res = jnp.zeros(ND // 2 + 1, dtype=jnp.complex128)

    # Unpacking for m == 0 and m == Nf cases:
    i_ind_range = jnp.arange(Nt // 2)
    i_0 = jnp.abs(i_ind_range)  # for m == 0: i = i_ind_range
    i_Nf = jnp.abs(Nf * (Nt // 2) - i_ind_range)
    ind3_0 = (2 * i_0) % Nt
    ind3_Nf = (2 * i_Nf) % Nt

    res = res.at[i_0].add(fft_prefactor2s[0, ind3_0] * phif[i_ind_range])
    res = res.at[i_Nf].add(fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range])
    # Special case for m == Nf (ensure the Nyquist frequency is updated correctly)
    special_index = jnp.abs(Nf * (Nt // 2) - (Nt // 2))
    res = res.at[special_index].add(fft_prefactor2s[Nf, 0] * phif[Nt // 2])

    # Unpacking for m in (1, ..., Nf-1)
    m_mid = m_range[1:Nf]
    # Use range [0, Nt//2) to match the loop in NumPy version.
    i_ind_range_mid = jnp.arange(Nt // 2)
    # Create meshgrid for vectorized computation.
    m_grid_mid, i_ind_grid_mid = jnp.meshgrid(
        m_mid, i_ind_range_mid, indexing="ij"
    )

    # Compute indices i1 and i2 following the NumPy logic.
    i1 = (Nt // 2) * m_grid_mid - i_ind_grid_mid
    i2 = (Nt // 2) * m_grid_mid + i_ind_grid_mid
    # Compute the wrapped indices for FFT results.
    ind31 = i1 % Nt
    ind32 = i2 % Nt

    # Update result array using vectorized adds.
    # Note: You might need to adjust this further if your target res shape is non-trivial,
    # because here we assume that i1 and i2 indices fall within the allocated result shape.
    res = res.at[i1].add(
        fft_prefactor2s[m_grid_mid, ind31] * phif[i_ind_grid_mid]
    )
    res = res.at[i2].add(
        fft_prefactor2s[m_grid_mid, ind32] * phif[i_ind_grid_mid]
    )

    return res


def phitilde_vec_norm(Nf: int, Nt: int, d: float) -> np.ndarray:
    """Normalize phitilde for inverse frequency domain transform."""
    df = 2 * np.pi / (Nf * Nt)
    omega = df * np.arange(0, Nt // 2 + 1, dtype=np.float64)

    dF = 1.0 / (2 * Nf)  # NOTE: missing 1/dt?
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


