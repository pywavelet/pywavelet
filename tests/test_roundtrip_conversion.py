import matplotlib.pyplot as plt
import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    plot_residuals,
    plot_time_domain_signal,
    plot_wavelet_domain_signal,
)
from pywavelet.transforms.types import TimeSeries
from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time

from scipy.signal.windows import tukey

def test_time_to_wavelet_to_time(make_plots, plot_dir):
    dt = 1 / 512
    Nt = 64
    Nf = 64
    mult = 16

    freq_range = [0.1, 50]
    assert max(freq_range) < 1 / (2 * dt), "Nyquist frequency is too low for the given frequency range"

    ND = Nt * Nf

    # time and frequency grids
    ts = np.arange(0, ND) * dt

    # generate signal
    h_time = generate_chirp_time_domain_signal(ts, freq_range)
    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt, mult=mult)
    h_reconstructed = from_wavelet_to_time(h_wavelet, mult=mult, dt=dt)

    if make_plots:
        fig = plot_time_domain_signal(h_time, freq_range)
        fig.savefig(f"{plot_dir}/original_signal.png", dpi=300)

        fig = h_wavelet.plot()
        fig.savefig(f"{plot_dir}/wavelet_domain.png", dpi=300)

        fig = plot_time_domain_signal(h_time, freq_range)
        fig.savefig(f"{plot_dir}/reconstructed_signal.png", dpi=300)

        fig = plot_residuals(h_time.data - h_reconstructed.data)
        fig.savefig(f"{plot_dir}/residuals.png", dpi=300)

    # check that the reconstructed signal is the same as the original
    residuals = h_time.data - h_reconstructed.data
    plt.figure()
    plt.plot(h_time.time, residuals)
    plt.show()

    assert np.abs(residuals).max() < 10

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2 ** pow_2) - N)), "constant")

def test_sine_wave_conversion(make_plots, plot_dir):
    dt = 1 / 512
    Nt = 1
    Nf = 64
    mult = 32

    ND = Nt * Nf

    # time and frequency grids
    ts = np.arange(0, ND) * dt

    # generate signal
    f_true = 10
    freq_range = [f_true - 5, f_true + 5]
    h_time = np.sin(2 * np.pi * f_true * ts)

    window = tukey(ND, 0.2)
    h_time = zero_pad(h_time * window)
    h_time = TimeSeries(h_time, time=ts)

    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt, mult=mult)
    h_reconstructed = from_wavelet_to_time(h_wavelet, mult=mult, dt=dt)

    if make_plots:

        fig = plot_time_domain_signal(h_time, freq_range)
        fig.savefig(f"{plot_dir}/original_signal.png", dpi=300)

        fig = h_wavelet.plot()
        fig.savefig(f"{plot_dir}/wavelet_domain.png", dpi=300)

        fig = plot_time_domain_signal(h_time, freq_range)
        fig.savefig(f"{plot_dir}/reconstructed_signal.png", dpi=300)

        fig = plot_residuals(h_time.data - h_reconstructed.data)
        fig.savefig(f"{plot_dir}/residuals.png", dpi=300)

    # check that the reconstructed signal is the same as the original
    residuals = h_time.data - h_reconstructed.data

    plt.figure()
    plt.plot(h_time.time, residuals)
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.show()

    assert np.abs(residuals).max() < 10
