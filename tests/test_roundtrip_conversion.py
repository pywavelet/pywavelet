import numpy as np
from utils import (
    plot_wavelet_comparison,
    plot_freqdomain_comparisions,
    plot_timedomain_comparisons
)

from pywavelet.transforms.types import Wavelet
from pywavelet.transforms import (
    from_wavelet_to_freq, from_wavelet_to_time,
    from_freq_to_wavelet, from_time_to_wavelet
)

from conftest import Nt, mult, dt, Nf, DATA_DIR


def test_timedomain_chirp_roundtrip(plot_dir, chirp_time):
    __run_timedomain_checks(chirp_time, "roundtrip_chirp_time", plot_dir)


def test_timedomain_sine_roundtrip(plot_dir, sine_time):
    __run_timedomain_checks(sine_time, "roundtrip_sine_time", plot_dir)


def test_freqdomain_chirp_roundtrip(plot_dir, chirp_freq):
    __run_freqdomain_checks(chirp_freq, "roundtrip_chirp_freq", plot_dir)


def test_freqdomain_sine_roundtrip(plot_dir, sine_freq):
    __run_freqdomain_checks(sine_freq, "roundtrip_sine_freq", plot_dir)


def __run_freqdomain_checks(hf, label, outdir):
    wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    __assert_wavelet_matches_cached_wavelet(wavelet, label, outdir)
    h_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)
    plot_freqdomain_comparisions(hf, h_reconstructed, wavelet, f"{outdir}/{label}.png")
    __assert_roundtrip_valid(hf, h_reconstructed, wavelet)


def __run_timedomain_checks(ht, label, outdir):
    wavelet = from_time_to_wavelet(ht, Nt=Nt, mult=mult)
    __assert_wavelet_matches_cached_wavelet(wavelet, label, outdir)
    h_reconstructed = from_wavelet_to_time(wavelet, mult=mult, dt=dt)
    plot_timedomain_comparisons(ht, h_reconstructed, wavelet, f"{outdir}/{label}.png")
    __assert_roundtrip_valid(ht, h_reconstructed, wavelet)


def __assert_roundtrip_valid(h_old, h_new, wavelet):
    minf, maxf = wavelet.freq[1], wavelet.freq[-1] / 2
    residuals = np.abs(h_old.data - h_new.data)
    if hasattr(h_old, "freq"):
        freq_mask = (minf < h_old.freq) & (h_old.freq < maxf)
        residuals = residuals[freq_mask]
    mean, std = np.mean(residuals), np.std(residuals)
    assert mean < 1e-3, f"Mean residual is too large: {mean}"
    assert std < 1e-3, f"Standard deviation of residuals is too large: {std}"
    assert np.max(np.abs(residuals)) < 1e-2, f"Max residual is too large: {np.max(np.abs(residuals))}"
    assert not np.isnan(residuals).any(), "Residuals contain NaNs"
    assert len(h_new.data) == len(h_old.data) == wavelet.ND


def __assert_wavelet_matches_cached_wavelet(cur: Wavelet, label, outdir):
    np.savez(f"{outdir}/{label}.npz", freq=cur.freq, time=cur.time, data=cur.data)
    cached_data = np.load(f"{DATA_DIR}/{label}.npz")
    cached = Wavelet(data=cached_data["data"], freq=cached_data["freq"], time=cached_data["time"])
    err = Wavelet(data=(cached.data - cur.data), freq=cur.freq, time=cur.time)
    net_err = np.sum(np.abs(err.data))

    plot_wavelet_comparison(cur, cached, err, label, outdir)

    assert net_err < 1e-3, f"Net error is too large: {net_err}"
    assert cur.__repr__() == cached.__repr__()
    assert cur.shape == cached.shape, f"Wavelets dont match current: {cur}, old: {cached}"
    assert np.allclose(cur.freq, cached.freq), f"Freqs dont match current: {cur.freq}, old: {cached.freq}"
    assert np.allclose(cur.time, cached.time), f"Times dont match current: {cur.time}, old: {cached.time}"
    assert np.allclose(cur.data, cached.data), f"Data doesnt match current: {cur.data}, old: {cached.data}"
