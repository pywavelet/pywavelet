from typing import Tuple

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gw_utils import DT, DURATION, get_ifo, inject_signal_in_noise
from matplotlib import colors

from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.utils.snr import compute_snr
from pywavelet.utils.lisa import get_lisa_data




Nf, Nt = 64, 64
ND = Nf * Nt
T_BINWIDTH = DURATION / Nt
F_BINWIDTH = 1 / 2 * T_BINWIDTH
FMAX = 1 / (2 * DT)

T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, FMAX, Nf)


def test_snr_lvk(plot_dir):
    distance = 10
    h_time, timeseries_snr = inject_signal_in_noise(
        mc=30, q=1, distance=distance, noise=False
    )
    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt)
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=ifo.power_spectral_density.psd_array,
        psd_f=ifo.power_spectral_density.frequency_array,
        f_grid=h_wavelet.freq.data,
        t_grid=h_wavelet.time.data,
    )
    wavelet_snr = compute_snr(h_wavelet, psd_wavelet)
    assert wavelet_snr == timeseries_snr


def test_lisa_snr(plot_dir):
    h_signal_t, t, h_signal_f, f, psd_f, snr = get_lisa_data()
    N = len(h_signal_t)
    Nt = 512
    Nf = N // Nt
    print(Nf, Nt, N)

    h_time = TimeSeries(data=h_signal_t, time=t)
    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd_f, psd_f=f, f_grid=h_wavelet.freq.data, t_grid=h_wavelet.time.data
    )
    wavelet_snr = compute_snr(h_wavelet, psd_wavelet)
    assert wavelet_snr  == snr ** 2
