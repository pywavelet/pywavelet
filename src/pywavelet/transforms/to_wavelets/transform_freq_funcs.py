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


def transform_wavelet_freq_helper(
    data: np.ndarray, Nf: int, Nt: int, phif: np.ndarray
) -> np.ndarray:
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt, dtype=np.complex128)
    freq_strain = data.data # Convert 
    for freq_bin in range(0, Nf + 1):
        __DX_assign_and_unpack_loop(freq_bin, Nt, Nf, DX, freq_strain, phif, wave)
    return wave

@njit
def __DX_assign_and_unpack_loop(
    freq_bin: int, Nt: int, Nf: int, DX: np.ndarray, data: np.ndarray, phif: np.ndarray, wave: np.ndarray
) -> None:
    """Helper for assigning DX and unpacking in the main loop"""
    i_base = Nt // 2
    jj_base = freq_bin * Nt // 2
    t_midpoint_bin = Nt // 2 

    if freq_bin == 0 or freq_bin == Nf:
        # first and last freq has special 'time-bin' midpoint
        DX[t_midpoint_bin] = phif[0] * data[freq_bin * t_midpoint_bin] / 2.0
    else:
        DX[t_midpoint_bin] = phif[0] * data[freq_bin * t_midpoint_bin]

    for jj in range(jj_base + 1 - Nt // 2, jj_base + Nt // 2):
        j = np.abs(jj - jj_base)
        i = i_base - jj_base + jj
        if freq_bin == Nf and jj > jj_base:
            DX[i] = 0.0
        elif freq_bin == 0 and jj < jj_base:
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]

    DX_trans = ifft(DX, Nt)

    if freq_bin == 0:
        for n in range(0, Nt, 2):
            wave[n, 0] = np.real(DX_trans[n] * np.sqrt(2))
    elif freq_bin == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = np.real(DX_trans[n] * np.sqrt(2))
    else:
        for n in range(0, Nt):
            if freq_bin % 2:
                if (n + freq_bin) % 2:
                    wave[n, freq_bin] = -np.imag(DX_trans[n])
                else:
                    wave[n, freq_bin] = np.real(DX_trans[n])
            else:
                if (n + freq_bin) % 2:
                    wave[n, freq_bin] = np.imag(DX_trans[n])
                else:
                    wave[n, freq_bin] = np.real(DX_trans[n])
