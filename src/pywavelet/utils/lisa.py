from typing import Tuple

import numpy as np
from scipy.signal.windows import tukey

from ..transforms.types import FrequencySeries, TimeSeries, Wavelet
from .snr import compute_frequency_optimal_snr


def lisa_psd_func(f):
    """
    PSD obtained from:
    Robson et al 2018, "LISA Sensitivity Curves"
    https://arxiv.org/pdf/1803.01944.pdf

    Removed galactic confusion noise. Non stationary effect.

    The power spectrum -- not the TDI

    """

    L = 2.5 * 10**9  # Length of LISA arm
    f0 = 19.09 * 10**-3

    # Eq 10
    Poms = ((1.5 * 10**-11) ** 2) * (
        1 + ((2 * 10**-3) / f) ** 4
    )  # Optical Metrology Sensor

    # Eq 11
    Pacc = (
        (3 * 10**-15) ** 2
        * (1 + (4 * 10**-3 / (10 * f)) ** 2)
        * (1 + (f / (8 * 10**-3)) ** 4)
    )  # Acceleration Noise

    # Eq 13
    PSD = (
        (10 / (3 * L**2))
        * (Poms + (4 * Pacc) / ((2 * np.pi * f)) ** 4)
        * (1 + 0.6 * (f / f0) ** 2)
    )  # PSD

    return PSD


def waveform(a: float, f: float, fdot: float, t: np.ndarray, eps=0):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.

    h = a * sin(2 * pi * (ft + 0.5 * fdot * t^2))
    Quadratic chirp signal.

    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))


def get_lisa_data():
    """
    This function is used to generate the data for the LISA detector. We use the waveform function to generate
    the signal and then use the freq_PSD function to generate the PSD. We then use the FFT function to generate
    the frequency domain waveform. We then compute the optimal SNR.
    """

    a_true = 5e-21
    f_true = 1e-3
    fdot_true = 1e-8

    fs = 2 * f_true  # Sampling rate
    delta_t = np.floor(
        0.01 / fs
    )  # Sampling interval -- largely oversampling here.
    tmax = 120 * 60 * 60  # 120 hours
    t = np.arange(0, tmax, delta_t)
    ND = int(
        2 ** (np.ceil(np.log2(len(t))))
    )  # Round length of time series to a power of two.
    t = np.arange(0, ND) * delta_t

    h_signal_t = waveform(a_true, f_true, fdot_true, t)
    freq = np.fft.fftfreq(ND, delta_t)[: ND // 2]
    psd_f = lisa_psd_func(freq)
    h_signal_f = np.fft.fft(h_signal_t)[: ND // 2]
    duration = delta_t * ND

    # skip first element to avoid division by zero
    freq = freq[1:]
    psd_f = psd_f[1:]
    h_signal_f = h_signal_f[1:]

    snr = compute_frequency_optimal_snr(h_signal_f, psd_f, duration)
    return h_signal_t, t, h_signal_f, h_signal_f, psd_f, snr
