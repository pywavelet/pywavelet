import matplotlib.pyplot as plt
from numpy.ma.core import absolute

from pywavelet.transforms.types import Wavelet
from pywavelet.transforms import (
    from_wavelet_to_freq, from_wavelet_to_time,
    from_freq_to_wavelet, from_time_to_wavelet
)
from pywavelet.transforms.types.plotting import plot_wavelet_trend
import numpy as np
from conftest import Nt, mult, dt, Nf, DATA_DIR


def test_plot(chirp_freq, plot_dir):
    wavelet = from_freq_to_wavelet(chirp_freq, Nf=Nf)
    freq_range = (0, 70)


    # chuck nans from t = wavelet.duration * 0.45 to wavelet.duration * 0.55
    wavelet.data[:, int(wavelet.Nt * 0.45):int(wavelet.Nt * 0.55)] = np.nan

    fig, ax = wavelet.plot(freq_range=freq_range, nan_color="white")
    wavelet.plot_trend(ax=ax, freq_range=freq_range)
    fig.savefig(f"{plot_dir}/test_wavelet_plot.png")
