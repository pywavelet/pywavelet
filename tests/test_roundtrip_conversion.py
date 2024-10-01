import os

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    plot_residuals,
)

from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms import from_wavelet_to_freq, from_wavelet_to_time, from_freq_to_wavelet, from_time_to_wavelet

from conftest import Nt, mult, dt, Nf


def test_timedomain_chirp_roundtrip(plot_dir, chirp_time):
    __run_timedomain_checks(
        chirp_time,
        f"{plot_dir}/out_roundtrip/chirp_time.png",
    )


def test_timedomain_sine_roundtrip(plot_dir, sine_time):
    __run_timedomain_checks(
        sine_time,
        f"{plot_dir}/out_roundtrip/sine_time.png",
    )


def test_freqdomain_chirp_roundtrip(plot_dir, chirp_freq):
    __run_freqdomain_checks(
        chirp_freq,
        f"{plot_dir}/out_roundtrip/chirp_freq.png",
    )


def test_freqdomain_sine_roundtrip(plot_dir, sine_freq):
    __run_freqdomain_checks(
        sine_freq,
        f"{plot_dir}/out_roundtrip/sine_freq.png",
    )


def __run_freqdomain_checks(hf, fname):
    ND = len(hf)
    _Nt = ND // Nf
    wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    assert wavelet.__repr__() == f"Wavelet(NfxNt={Nf}x{_Nt})"
    assert len(wavelet.freq) == Nf
    assert len(wavelet.time) == _Nt
    h_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)
    assert len(h_reconstructed.data) == len(hf.data) == wavelet.ND
    assert not np.isnan(h_reconstructed.data).any(), "Reconstructed data contains NaNs"
    _make_freqdomain_plots(hf, h_reconstructed, wavelet, fname)


def __run_timedomain_checks(ht, fname):
    wavelet = from_time_to_wavelet(ht, Nt=Nt, mult=mult)
    assert wavelet.__repr__() == f"Wavelet(NfxNt={Nf}x{Nt})"
    assert len(wavelet.freq) == Nf
    assert len(wavelet.time) == Nt
    h_reconstructed = from_wavelet_to_time(wavelet, mult=mult, dt=dt)
    assert len(h_reconstructed.data) == len(ht.data) == wavelet.ND
    assert not np.isnan(h_reconstructed.data).any(), "Reconstructed data contains NaNs"
    _make_timedomain_plots(ht, h_reconstructed, wavelet, fname)



def _make_timedomain_plots(ht: TimeSeries, h_reconstructed, wavelet, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ht.plot(ax=axes[0], label="Original")
    h_reconstructed.plot(ax=axes[0], label="Reconstructed", linestyle="--", color="tab:orange", alpha=0.5)
    axes[0].legend()
    wavelet.plot(ax=axes[1])
    r = ht.data - h_reconstructed.data
    plot_residuals(ht.data - h_reconstructed.data, axes[2])
    axes[0].set_title("Timeseries")
    axes[1].set_title("Wavelet")
    axes[2].set_title("Residuals")
    plt.tight_layout()
    plt.savefig(fname)

    assert np.mean(r) < 1e-3, "Mean residual is too large"
    assert np.std(r) < 1e-3, "Standard deviation of residuals is too large"
    assert np.max(np.abs(r)) < 1e-2, "Max residual is too large"

def _make_freqdomain_plots(hf: FrequencySeries, h_reconstructed, wavelet, fname):
    minf, maxf = wavelet.freq[1], wavelet.freq[-1] / 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hf.plot_periodogram(ax=axes[0], label="Original")
    h_reconstructed.plot_periodogram(ax=axes[0], label="Reconstructed", linestyle="--", color="tab:orange", alpha=0.5)
    axes[0].axvline(hf.nyquist_frequency, linestyle="--", color="tab:red", label="Nyquist frequency")
    axes[0].axvline(maxf, linestyle="--", color="tab:green", label="wavelet max-min f")
    axes[0].axvline(minf, linestyle="--", color="tab:green")
    axes[0].legend()
    wavelet.plot(ax=axes[1])
    freq_mask = (minf < hf.freq) & (hf.freq < maxf)
    r = (np.abs(hf.data) - np.abs(h_reconstructed.data))[freq_mask]
    plot_residuals(r, axes[2])
    axes[2].set_title("Residuals (in WDF f-range)")
    axes[0].set_title("Periodogram")
    axes[1].set_title("Wavelet")
    plt.tight_layout()
    plt.savefig(fname)

    assert np.mean(r) < 1e-3, "Mean residual is too large"
    assert np.std(r) < 1e-3, "Standard deviation of residuals is too large"
    assert np.max(np.abs(r)) < 1e-2, "Max residual is too large"
