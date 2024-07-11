import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    plot_residuals,
)

from pywavelet.data import Data, TimeSeries
from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time

dt = 1 / 512
Nt = 64
Nf = 64
mult = 16
ND = Nt * Nf
ts = np.arange(0, ND) * dt


def test_time_to_wavelet_to_time(make_plots, plot_dir):
    freq_range = [20, 100]
    h_time = generate_chirp_time_domain_signal(ts, freq_range)
    __run_checks(
        h_time,
        Nt,
        mult,
        dt,
        freq_range,
        make_plots,
        f"{plot_dir}/out_roundtrip/chirp.png",
    )


def test_sine_wave_conversion(make_plots, plot_dir):
    f_true = 10
    freq_range = [f_true - 5, f_true + 5]
    h_time = generate_sine_time_domain_signal(ts, ND, f_true=f_true)
    __run_checks(
        h_time,
        Nt,
        mult,
        dt,
        freq_range,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine.png",
    )


def __run_checks(h_time, Nt, mult, dt, freq_range, make_plots, fname):
    data = Data.from_timeseries(
        h_time,
        Nt=Nt,
        mult=mult,
        minimum_frequency=freq_range[0],
        maximum_frequency=freq_range[1],
    )
    h_reconstructed = from_wavelet_to_time(data.wavelet, mult=mult, dt=dt)

    if make_plots:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        __make_plots(h_time, h_reconstructed, data, fname=fname)

    residuals = h_time.data - h_reconstructed.data
    mean, std = residuals.mean(), residuals.std()
    assert np.abs(mean) < 0.1
    assert np.abs(std) < 1


def __make_plots(h_time, h_reconstructed, data, fname):
    fig, axes = plt.subplots(5, 1, figsize=(4, 14))
    data.plot_all(axes=axes)
    plot_residuals(h_time - h_reconstructed, ax=axes[4])
    plt.tight_layout()
    fig.savefig(fname, dpi=300)
