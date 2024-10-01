from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

from pywavelet.transforms.types import TimeSeries, FrequencySeries


def __symlog_bins(data, lin_thresh=None, log_base=10, num_lin=5, num_log=10):
    if lin_thresh is None:
        lin_thresh = 0.05 * np.std(data)
    lin_bins = np.linspace(-lin_thresh, lin_thresh, num_lin)
    log_bins_pos = np.logspace(np.log10(lin_thresh), np.log10(np.max(np.abs(data))), num=num_log, base=log_base)
    log_bins_neg = -np.logspace(np.log10(lin_thresh), np.log10(np.max(np.abs(data))), num=num_log, base=log_base)[::-1]
    bins = np.concatenate([log_bins_neg, lin_bins, log_bins_pos])
    bins = np.sort(np.unique(bins))
    return bins, lin_thresh


def plot_residuals(residuals, ax: plt.Axes, symlog=True):
    mean, std = np.mean(residuals), np.std(residuals)
    ax.text(
        0.05,
        0.95,
        r"${:.2f} \pm {:.2f}$".format(mean, std),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    num_nans = np.sum(np.isnan(residuals))
    assert num_nans == 0, f"Found {num_nans} NaNs in residuals."
    if symlog:
        bins, lin_thresh = __symlog_bins(residuals)
        ax.hist(residuals, bins=bins, density=False)
        ax.set_xscale("symlog", linthresh=lin_thresh)
    else:
        ax.hist(residuals, bins=100, density=False)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")


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
    return TimeSeries(data=y, time=t)


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2 ** pow_2) - N)), "constant")


def generate_sine_time_domain_signal(ts, n, f_true=10):
    h_signal = np.sin(2 * np.pi * f_true * ts)
    window = tukey(n, 0.0)
    h_signal = __zero_pad(h_signal * window)
    return TimeSeries(h_signal, time=ts)


def generate_sine_freq_domain_signal(fs, f_true=10):
    freq = np.fft.rfftfreq(fs, d=1 / fs)
    hf = np.zeros_like(freq)
    # closest frequency bin to f_true
    idx = np.argmin(np.abs(freq - f_true))
    hf[idx] = 1 / fs
    return FrequencySeries(data=hf, freq=freq)
