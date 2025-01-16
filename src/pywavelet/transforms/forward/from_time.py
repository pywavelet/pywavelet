"""helper functions for transform_time.py"""

import numpy as np
from numba import njit
from numpy import fft


def transform_wavelet_time_helper(
    data: np.ndarray, Nf: int, Nt: int, phi: np.ndarray, mult: int
) -> np.ndarray:
    """helper function to do the wavelet transform in the time domain"""
    # the time domain freqseries stream
    ND = Nf * Nt
    K = mult * 2 * Nf
    assert len(data) == ND, f"len(data)={len(data)} != Nf*Nt={ND}"

    # windowed freqseries packets
    wdata = np.zeros(K)
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal
    data_pad = np.concatenate((data, data[:K]))
    __core(Nf, Nt, K, ND, wdata, data_pad, phi, wave, mult)
    return wave


def __core(
    Nf: int,
    Nt: int,
    K: int,
    ND: int,
    wdata: np.ndarray,
    data_pad: np.ndarray,
    phi: np.ndarray,
    wave: np.ndarray,
    mult: int,
) -> None:
    for time_bin_i in range(0, Nt):
        __fill_wave_1(time_bin_i, K, ND, Nf, wdata, data_pad, phi)
        wdata_trans = np.fft.rfft(wdata, K)
        __fill_wave_2(time_bin_i, wave, wdata_trans, Nf, mult)


@njit()
def __fill_wave_1(
    t_bin: int,
    K: int,
    ND: int,
    Nf: int,
    wdata: np.ndarray,
    data_pad: np.ndarray,
    phi: np.ndarray,
) -> None:
    """Assign wdata to be FFT'd in a loop with K extra values on the right to loop."""
    # wrapping the freqseries is needed to make the sum in Eq 13 in Cornish paper from [-K/2, K/2]
    jj = (t_bin * Nf - K // 2) % ND  # Periodically wrap the freqseries
    for j in range(K):
        # Eq 13 from Cornish paper
        wdata[j] = data_pad[jj] * phi[j]  # Apply the window
        jj = (jj + 1) % ND  # Periodically wrap the freqseries


@njit()
def __fill_wave_2(
    t_bin: int, wave: np.ndarray, wdata_trans: np.ndarray, Nf: int, mult: int
) -> None:
    # wdata_trans = np.sum(wdata) * np.exp(1j * np.pi * np.arange(0, 1+K//2) / K)

    # pack fft'd wdata into wave array
    if t_bin % 2 == 0 and t_bin < wave.shape[0] - 1:  # if EVEN t_bin
        # m=0 value at even Nt and
        wave[t_bin, 0] = wdata_trans[0].real / np.sqrt(2)
        wave[t_bin + 1, 0] = wdata_trans[Nf * mult].real / np.sqrt(2)

    # Cnm in eq 13
    for j in range(1, Nf):
        if (t_bin + j) % 2:
            wave[t_bin, j] = -wdata_trans[j * mult].imag
        else:
            wave[t_bin, j] = wdata_trans[j * mult].real
