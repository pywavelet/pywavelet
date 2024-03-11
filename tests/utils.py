from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.waveform import get_td_waveform
from scipy.signal import chirp, spectrogram
from scipy.signal.windows import tukey

from pywavelet.transforms.types import TimeAxis, TimeSeries


def cbc_waveform(mc, q=1, delta_t=1.0 / 4096, f_lower=20):
    m1 = mass1_from_mchirp_q(mc, q)
    m2 = mass2_from_mchirp_q(mc, q)
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        delta_t=delta_t,
        f_lower=f_lower,
    )
    data = TimeSeries(hp.data, time=TimeAxis(hp.sample_times.data))
    return data


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


# plot signal
def plot_time_domain_signal(signal: TimeSeries, freq_range: List[float]):
    t = signal.time
    h = signal.data
    T = max(signal.time)
    fs = 1 / (t[1] - t[0])
    ff, tt, Sxx = spectrogram(h, fs=fs, nperseg=256, nfft=576)
    freq, h_freq = waveform_fft(t, h)

    fig, axes = plt.subplots(3, 1, figsize=(4, 6))
    axes[0].plot(t, h, lw=0.1)
    axes[0].set_ylabel("h(t)")
    axes[0].set_xlim(0, T)
    axes[1].plot(freq, np.abs(h_freq))
    axes[1].set_ylabel("|h(f)|")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(*freq_range)
    axes[2].set_xlim(0, T)
    axes[2].pcolormesh(tt, ff, Sxx)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlim(0, T)
    axes[2].set_ylim(*freq_range)
    # add colorbar to the last axis
    cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
    cbar.set_label("Amplitude")
    plt.tight_layout()
    return fig


def plot_wavelet_domain_signal(wavelet_data, time_grid, freq_grid, freq_range):
    fig = plt.figure()
    plt.imshow(
        np.abs(np.rot90(wavelet_data)),
        aspect="auto",
        extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
    )
    cbar = plt.colorbar()
    cbar.set_label("Wavelet Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(*freq_range)
    plt.tight_layout()
    return fig


def plot_residuals(residuals):
    fig = plt.figure()
    plt.hist(residuals, bins=100)
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    return fig


def generate_chirp_time_domain_signal(
    t: np.ndarray, freq_range: List[float]
) -> TimeSeries:
    y = chirp(
        t, f0=freq_range[0], f1=freq_range[1], t1=t[-1], method="quadratic"
    )
    return TimeSeries(data=y, time=TimeAxis(t))
