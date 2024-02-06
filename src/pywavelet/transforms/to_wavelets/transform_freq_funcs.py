"""helper functions for transform_freq"""
import numpy as np
from numba import njit

from pywavelet import fft_funcs as fft


@njit()
def tukey(data: np.ndarray, alpha: float, N: int) -> None:
    """apply tukey window function to data"""
    imin = np.int64(alpha * (N - 1) / 2)
    imax = np.int64((N - 1) * (1 - alpha / 2))
    Nwin = N - imax

    for i in range(0, N):
        f_mult = 1.0
        if i < imin:
            f_mult = 0.5 * (1.0 + np.cos(np.pi * (i / imin - 1.0)))
        if i > imax:
            f_mult = 0.5 * (1.0 + np.cos(np.pi / Nwin * (i - imax)))
        data[i] *= f_mult


def transform_wavelet_freq_helper(data: np.ndarray, Nf: int, Nt: int, phif: np.ndarray) -> np.ndarray:
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt, dtype=np.complex128)
    for m in range(0, Nf + 1):
        __DX_assign_loop(m, Nt, Nf, DX, data, phif)
        DX_trans = fft.ifft(DX, Nt)
        __DX_unpack_loop(m, Nt, Nf, DX_trans, wave)
    return wave


@njit()
def __DX_assign_loop(m: int, Nt: int, Nf: int, DX: np.ndarray, data: np.ndarray, phif: np.ndarray) -> None:
    """helper for assigning DX in the main loop"""
    i_base = Nt // 2
    jj_base = m * Nt // 2

    if m == 0 or m == Nf:
        # NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
        DX[Nt // 2] = phif[0] * data[m * Nt // 2] / 2.0
        DX[Nt // 2] = phif[0] * data[m * Nt // 2] / 2.0
    else:
        DX[Nt // 2] = phif[0] * data[m * Nt // 2]
        DX[Nt // 2] = phif[0] * data[m * Nt // 2]

    for jj in range(jj_base + 1 - Nt // 2, jj_base + Nt // 2):
        j = np.abs(jj - jj_base)
        i = i_base - jj_base + jj
        if m == Nf and jj > jj_base:
            DX[i] = 0.0
        elif m == 0 and jj < jj_base:
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]


@njit()
def __DX_unpack_loop(m: int, Nt: int, Nf: int, DX_trans: np.ndarray, wave: np.ndarray) -> None:
    """helper for unpacking fftd DX in main loop"""
    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of m=0 respectively
        for n in range(0, Nt, 2):
            wave[n, 0] = np.real(DX_trans[n] * np.sqrt(2))
    elif m == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = np.real(DX_trans[n] * np.sqrt(2))
    else:
        for n in range(0, Nt):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = -np.imag(DX_trans[n])
                else:
                    wave[n, m] = np.real(DX_trans[n])
            else:
                if (n + m) % 2:
                    wave[n, m] = np.imag(DX_trans[n])
                else:
                    wave[n, m] = np.real(DX_trans[n])
