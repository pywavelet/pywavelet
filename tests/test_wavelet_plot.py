import numpy as np
from conftest import Nf, Nt

from pywavelet.types import Wavelet


def test_plot(chirp_freq, plot_dir):
    chirp = chirp_freq.to_timeseries()
    chirp = chirp.zero_pad_to_power_of_2(tukey_window_alpha=0.2)
    wavelet = chirp.to_wavelet(Nf=Nf, Nt=Nt)
    freq_range = (0, 70)

    # chuck nans from t = wavelet.duration * 0.45 to wavelet.duration * 0.55
    wavelet.data[:, int(wavelet.Nt * 0.45) : int(wavelet.Nt * 0.55)] = np.nan

    fig, ax = wavelet.plot(freq_range=freq_range, nan_color="white")
    wavelet.plot_trend(ax=ax, freq_range=freq_range)
    fig.savefig(f"{plot_dir}/test_wavelet_plot.png")

    w = Wavelet.zeros(32, 32, 1)
    w.data[:, 16] = 1
    fig, ax = w.plot()
    fig.savefig(f"{plot_dir}/test_wavelet_plot_zeros.png")
