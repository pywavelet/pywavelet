"""functions for computing the inverse wavelet transforms"""

import numpy as np
from numba import njit
from numpy import fft


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
            val = wave_in[n, m]  # bug is here
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
