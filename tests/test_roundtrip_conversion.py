import logging
import os

import numpy as np
import pytest
from conftest import DATA_DIR, Nf, Nt, dt
from utils import (
    cuda_available,
    plot_freqdomain_comparisions,
    plot_timedomain_comparisons,
    plot_wavelet_comparison,
    to_cupy,
    to_jax,
    to_numpy,
)

from pywavelet import set_backend
from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

logger = logging.getLogger("pywavelet")


def test_timedomain_sine_roundtrip(plot_dir, sine_time):
    _run_timedomain_checks(sine_time, "roundtrip_sine_time", plot_dir)
    logger.info("-------test complete--------")


@pytest.mark.parametrize("backend", ["numpy", "jax", "cupy"])
def test_freqdomain_chirp_roundtrip(backend, plot_dir, chirp_freq):
    from pywavelet import set_backend

    set_backend(backend)
    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")
    _run_freqdomain_checks(chirp_freq, "roundtrip_chirp_freq", plot_dir)
    logger.info("-------test complete--------")


@pytest.mark.parametrize("backend", ["numpy", "jax", "cupy"])
def test_freqdomain_sine_roundtrip(backend, plot_dir, sine_freq):
    from pywavelet import set_backend

    set_backend(backend)
    if backend == "cupy":
        if not cuda_available:
            pytest.skip("CUDA is not available")
        else:
            sine_freq = to_cupy(sine_freq)

    elif backend == "jax":
        sine_freq = to_jax(sine_freq)

    _run_freqdomain_checks(sine_freq, "roundtrip_sine_freq", plot_dir)
    logger.info("-------test complete--------")


def test_conversion_from_hf_ht(sine_freq):
    set_backend("numpy")
    ht = sine_freq.to_timeseries()
    w1 = sine_freq.to_wavelet(Nf=Nf, Nt=Nt)
    w2 = ht.to_wavelet(Nf=Nf, Nt=Nt)
    assert w1 == w2
    hf1 = w1.to_frequencyseries()
    ht2 = w2.to_timeseries()
    assert sine_freq == hf1
    assert ht == ht2


def _run_freqdomain_checks(hf, label, outdir, Nf=Nf, dt=dt):
    from pywavelet.backend import current_backend, xp
    from pywavelet.transforms import from_freq_to_wavelet, from_wavelet_to_freq

    outdir = f"{outdir}/{label}"
    os.makedirs(outdir, exist_ok=True)

    wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    wavelet_np = to_numpy(wavelet)

    _assert_wavelet_matches_cached_wavelet(wavelet_np, label, outdir)

    h_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)
    h_reconstructed = to_numpy(h_reconstructed)

    plot_fn = f"{outdir}/{label}_{current_backend}.png"
    plot_freqdomain_comparisions(
        hf,
        h_reconstructed,
        wavelet,
        plot_fn,
    )
    _assert_roundtrip_valid(hf, h_reconstructed, wavelet)
    return h_reconstructed


def _run_timedomain_checks(ht, label, outdir, Nt=Nt, dt=dt):
    from pywavelet.backend import current_backend
    from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time

    outdir = f"{outdir}/{label}"
    os.makedirs(outdir, exist_ok=True)

    wavelet = from_time_to_wavelet(ht, Nt=Nt)
    _assert_wavelet_matches_cached_wavelet(wavelet, label, outdir)
    h_reconstructed = from_wavelet_to_time(wavelet, dt=dt)
    plot_fn = f"{outdir}/{label}_{current_backend}.png"
    plot_timedomain_comparisons(ht, h_reconstructed, wavelet, plot_fn)
    _assert_roundtrip_valid(ht, h_reconstructed, wavelet)


def _assert_roundtrip_valid(h_old, h_new, wavelet):
    h_new = to_numpy(h_new)
    h_old = to_numpy(h_old)
    wavelet = to_numpy(wavelet)

    residuals = np.abs(h_old.data - h_new.data)
    mean, std = np.mean(residuals), np.std(residuals)
    assert mean < 1e-3, f"Mean residual is too large: {mean}"
    assert std < 1e-3, f"Standard deviation of residuals is too large: {std}"
    assert (
        np.max(np.abs(residuals)) < 1e-2
    ), f"Max residual is too large: {np.max(np.abs(residuals))}"
    assert not np.isnan(residuals).any(), "Residuals contain NaNs"
    assert np.allclose(h_old.shape, h_new.shape)
    assert (
        h_old.ND == h_old.ND == wavelet.ND
    ), f"ND dont match: {h_old.ND}, {h_new.ND}, {wavelet.ND}"


def _assert_wavelet_matches_cached_wavelet(cur: "Wavelet", label, outdir):
    from pywavelet.backend import current_backend
    from pywavelet.types import Wavelet

    curr = to_numpy(cur)
    fig, ax = curr.plot()
    fig.savefig(f"{outdir}/{label}_{current_backend}_wavelet.png")

    np.savez(
        f"{outdir}/{label}.npz", freq=cur.freq, time=cur.time, data=cur.data
    )
    cached_data = np.load(f"{DATA_DIR}/{label}.npz")
    cached = Wavelet(
        data=cached_data["data"],
        freq=cached_data["freq"],
        time=cached_data["time"],
    )
    err = Wavelet(data=(cached.data - cur.data), freq=cur.freq, time=cur.time)
    net_err = np.sum(np.abs(err.data))

    label = f"{label}_{current_backend}"
    plot_wavelet_comparison(cur, cached, err, label, outdir)

    assert net_err < 0.9, f"Net error (orig - new WDM) is too large: {net_err}"
    assert (
        cur.__repr__() == cached.__repr__()
    ), f"Current[{cur.__repr__()}] != Old[{cached.__repr__()}]"
    assert (
        cur.shape == cached.shape
    ), f"Wavelets dont match current: {cur}, old: {cached}"
    assert np.allclose(
        cur.freq, cached.freq
    ), f"Freqs dont match current: {cur.freq}, old: {cached.freq}"
    assert np.allclose(
        cur.time, cached.time
    ), f"Times dont match current: {cur.time}, old: {cached.time}"
    assert np.allclose(
        cur.data, cached.data
    ), f"Data doesnt match current: {cur.data}, old: {cached.data}"
