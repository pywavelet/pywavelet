from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

from pywavelet.transforms.types import TimeAxis, TimeSeries


def waveform_fft(
    t,
    waveform,
):
    N = len(waveform)
    taper = tukey(N, 0.1)
    waveform_w_pad = zero_pad(waveform * taper)
    waveform_f = np.fft.rfft(waveform_w_pad)[1:]
    n_t = len(zero_pad(t))
    delta_t = t[1] - t[0]
    freq = np.fft.rfftfreq(n_t, delta_t)[1:]
    return freq, waveform_f


def zero_pad(data):
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def plot_wavelet_domain_signal(
    wavelet_data, time_grid, freq_grid, freq_range
) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(
        np.abs(np.rot90(wavelet_data)),
        aspect="auto",
        extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
    )
    cbar = plt.colorbar()
    cbar.set_label("Wavelet Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(*freq_range)
    plt.tight_layout()
    return fig, ax


def plot_residuals(residuals: TimeSeries, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes[0]
    ax.plot(residuals)
    ax.set_ylabel("Residuals")
    # horizontal line at y = 0
    ax.axhline(0, color="k", linestyle="--")
    # add txtbox with stats
    mean, std = np.mean(residuals), np.std(residuals)
    # stats txtbox in latex
    stats = r"err = ${:.2f} \pm {:.2f}$".format(mean, std)
    ax.text(
        0.05,
        0.95,
        stats,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    ax = axes[1]
    ax.hist(np.abs(residuals), bins=50, density=True)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")

    fig = axes[0].get_figure()

    return fig, ax


def generate_chirp_time_domain_signal(
    t: np.ndarray, freq_range: List[float]
) -> TimeSeries:
    fs = 1 / (t[1] - t[0])
    nyquist = fs / 2
    fmax = max(freq_range)
    assert (
        fmax < nyquist
    ), f"f_max [{fmax:.2f} Hz] must be less than f_nyquist [{nyquist:2f} Hz]."

    y = chirp(
        t, f0=freq_range[0], f1=freq_range[1], t1=t[-1], method="quadratic"
    )
    return TimeSeries(data=y, time=TimeAxis(t))


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def generate_sine_time_domain_signal(ts, n, f_true=10):
    h_signal = np.sin(2 * np.pi * f_true * ts)
    window = tukey(n, 0.0)
    h_signal = __zero_pad(h_signal * window)
    return TimeSeries(h_signal, time=ts)
