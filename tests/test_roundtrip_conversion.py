import matplotlib.pyplot as plt
import numpy as np
from utils import plot_residuals

from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.transforms import from_wavelet_to_freq, from_wavelet_to_time, from_freq_to_wavelet, from_time_to_wavelet

from matplotlib.colors import TwoSlopeNorm

from conftest import Nt, mult, dt, Nf, DATA_DIR, BRANCH


def test_timedomain_chirp_roundtrip(plot_dir, chirp_time):
    __run_timedomain_checks(chirp_time, "roundtrip_chirp_time", plot_dir)


def test_timedomain_sine_roundtrip(plot_dir, sine_time):
    __run_timedomain_checks(sine_time, "roundtrip_sine_time", plot_dir)


def test_freqdomain_chirp_roundtrip(plot_dir, chirp_freq):
    __run_freqdomain_checks(chirp_freq, "roundtrip_chirp_freq", plot_dir)


def test_freqdomain_sine_roundtrip(plot_dir, sine_freq):
    __run_freqdomain_checks(sine_freq, "roundtrip_sine_freq", plot_dir)


def __run_freqdomain_checks(hf, label, outdir):
    ND = len(hf)
    _Nt = ND // Nf
    wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    np.savez(f"{outdir}/{label}.npz", freq=wavelet.freq, time=wavelet.time, data=wavelet.data)
    __compare_wavelet_to_cached(wavelet, label, outdir)
    assert wavelet.__repr__() == f"Wavelet(NfxNt={Nf}x{_Nt})"
    assert len(wavelet.freq) == Nf
    assert len(wavelet.time) == _Nt
    h_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)
    assert len(h_reconstructed.data) == len(hf.data) == wavelet.ND
    assert not np.isnan(h_reconstructed.data).any(), "Reconstructed data contains NaNs"
    _make_freqdomain_plots(hf, h_reconstructed, wavelet, f"{outdir}/{label}.png")


def __run_timedomain_checks(ht, label, outdir):
    wavelet = from_time_to_wavelet(ht, Nt=Nt, mult=mult)
    np.savez(f"{outdir}/{label}.npz", freq=wavelet.freq, time=wavelet.time, data=wavelet.data)
    __compare_wavelet_to_cached(wavelet, label, outdir)
    assert wavelet.__repr__() == f"Wavelet(NfxNt={Nf}x{Nt})"
    assert len(wavelet.freq) == Nf
    assert len(wavelet.time) == Nt
    h_reconstructed = from_wavelet_to_time(wavelet, mult=mult, dt=dt)
    assert len(h_reconstructed.data) == len(ht.data) == wavelet.ND
    assert not np.isnan(h_reconstructed.data).any(), "Reconstructed data contains NaNs"
    _make_timedomain_plots(ht, h_reconstructed, wavelet, f"{outdir}/{label}.png")


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
    axes[0].axvline(maxf, linestyle="--", color="tab:green", label="cur max-min f")
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


def __compare_wavelet_to_cached(cur, label, outdir):
    cached_data = np.load(f"{DATA_DIR}/{label}.npz")
    cached = Wavelet(data=cached_data["data"], freq=cached_data["freq"], time=cached_data["time"])
    err = Wavelet(data=(cached.data - cur.data) / (cur.data), freq=cur.freq, time=cur.time)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].set_title(f"Branch: {BRANCH}")
    axes[1].set_title("Cached (v0.0.1)")
    axes[2].set_title("Relative error")
    vmin = min(np.min(cur.data), np.min(cached.data))
    vmax = max(np.max(cur.data), np.max(cached.data))

    if vmin == vmax:
        vmin = -1
        vmax = 1
    norm = TwoSlopeNorm(
        vmin=vmin, vcenter=0, vmax=vmax
    )

    cur.plot(ax=axes[0], norm=norm, cmap='bwr', show_colorbar=False)
    cached.plot(ax=axes[1], norm=norm, cmap='bwr', show_colorbar=False)
    err.plot(ax=axes[2], absolute=True, norm=norm, cmap='bwr', show_colorbar=True)
    plt.savefig(f"{outdir}/{label}_comparison.png")
    assert cur.shape == cached.shape, f"Wavelets dont match current: {cur}, old: {cached}"
    assert np.allclose(cur.freq, cached.freq), f"Freqs dont match current: {cur.freq}, old: {cached.freq}"
    assert np.allclose(cur.time, cached.time), f"Times dont match current: {cur.time}, old: {cached.time}"
    assert np.allclose(cur.data, cached.data), f"Data doesnt match current: {cur.data}, old: {cached.data}"
