"""helper functions for transform_time.py"""
import numpy as np
from numba import njit

from ... import fft_funcs as fft


def transform_wavelet_time_helper(
    data: np.ndarray, Nf: int, Nt: int, phi: np.ndarray, mult: int
) -> np.ndarray:
    """helper function to do the wavelet transform in the time domain"""
    # the time domain data stream
    ND = Nf * Nt
    K = mult * 2 * Nf
    assert len(data) == ND, f"len(data)={len(data)} != Nf*Nt={ND}"

    # windowed data packets
    wdata = np.zeros(K)
    wave = np.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal
    data_pad = np.concatenate((data, data[:K]))

    for i in range(0, Nt):
        __fill_wave(i, K, ND, Nf, wdata, data_pad, phi, mult, wave)

    return wave


def __fill_wave(
    t_bin: int,
    K: int,
    ND: int,
    Nf: int,
    wdata: np.ndarray,
    data_pad: np.ndarray,
    phi: np.ndarray,
    mult,
    wave,
) -> None:
    """Assign wdata to be FFT'd in a loop with K extra values on the right to loop."""
    # wrapping the data is needed to make the sum in Eq 13 in Cornish paper from [-K/2, K/2]
    jj = (t_bin * Nf - K // 2) % ND  # Periodically wrap the data
    for j in range(K):
        # Eq 13 from Cornish paper
        wdata[j] = data_pad[jj] * phi[j]  # Apply the window
        jj = (jj + 1) % ND  # Periodically wrap the data

    # rfft --> real part of the fft (0 to Nf)
    # FIXME: this breaks njit beacause numba doesn't support rfft
    wdata_trans = fft.rfft(wdata, K)

    # wdata_trans = np.sum(wdata) * np.exp(1j * np.pi * np.arange(0, 1+K//2) / K)

    # pack fft'd wdata into wave array
    if t_bin % 2 == 0 and t_bin < wave.shape[0] - 1:  # if EVEN t_bin
        # m=0 value at even Nt and
        wave[t_bin, 0] = np.real(wdata_trans[0]) / np.sqrt(2)
        wave[t_bin + 1, 0] = np.real(wdata_trans[Nf * mult]) / np.sqrt(2)

    # Cnm in eq 13
    for j in range(1, Nf):
        if (t_bin + j) % 2:
            wave[t_bin, j] = -np.imag(wdata_trans[j * mult])
        else:
            wave[t_bin, j] = np.real(wdata_trans[j * mult])
