"""helper functions for transform_freq"""

import numpy as np
from numba import njit


def transform_wavelet_freq_helper(
    data: np.ndarray, Nf: int, Nt: int, phif: np.ndarray
) -> np.ndarray:
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal
    DX = np.zeros(Nt, dtype=np.complex128)
    freq_strain = data.copy()  # Convert
    __core(Nf, Nt, DX, freq_strain, phif, wave)
    return wave


# @njit()
def __core(
    Nf: int,
    Nt: int,
    DX: np.ndarray,
    freq_strain: np.ndarray,
    phif: np.ndarray,
    wave: np.ndarray,
):
    for f_bin in range(0, Nf + 1):
        __fill_wave_1(f_bin, Nt, Nf, DX, freq_strain, phif)
        # Numba doesn't support np.ifft so we cant jit this
        DX_trans = np.fft.ifft(DX, Nt)
        __fill_wave_2(f_bin, DX_trans, wave, Nt, Nf)


@njit()
def __fill_wave_1(
    f_bin: int,
    Nt: int,
    Nf: int,
    DX: np.ndarray,
    data: np.ndarray,
    phif: np.ndarray,
) -> None:
    """helper for assigning DX in the main loop"""
    i_base = Nt // 2
    jj_base = f_bin * Nt // 2

    if f_bin == 0 or f_bin == Nf:
        # NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
        DX[Nt // 2] = phif[0] * data[f_bin * Nt // 2] / 2.0
    else:
        DX[Nt // 2] = phif[0] * data[f_bin * Nt // 2]

    for jj in range(jj_base + 1 - Nt // 2, jj_base + Nt // 2):
        j = np.abs(jj - jj_base)
        i = i_base - jj_base + jj
        if f_bin == Nf and jj > jj_base:
            DX[i] = 0.0
        elif f_bin == 0 and jj < jj_base:
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]


@njit()
def __fill_wave_2(
    f_bin: int, DX_trans: np.ndarray, wave: np.ndarray, Nt: int, Nf: int
) -> None:
    if f_bin == 0:
        # half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of f_bin=0 respectively
        for n in range(0, Nt, 2):
            wave[n, 0] = DX_trans[n].real * np.sqrt(2)
    elif f_bin == Nf:
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = DX_trans[n].real * np.sqrt(2)
    else:
        for n in range(0, Nt):
            if f_bin % 2:
                if (n + f_bin) % 2:
                    wave[n, f_bin] = -DX_trans[n].imag
                else:
                    wave[n, f_bin] = DX_trans[n].real
            else:
                if (n + f_bin) % 2:
                    wave[n, f_bin] = DX_trans[n].imag
                else:
                    wave[n, f_bin] = DX_trans[n].real
