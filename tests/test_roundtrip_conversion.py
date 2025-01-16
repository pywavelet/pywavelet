import numpy as np
from conftest import DATA_DIR, Nf, Nt, dt, mult
from utils import (
    generate_pure_f0,
    plot_fft,
    plot_freqdomain_comparisions,
    plot_timedomain_comparisons,
    plot_wavelet_comparison,
)

from pywavelet.transforms import (
    from_freq_to_wavelet,
    from_time_to_wavelet,
    from_wavelet_to_freq,
    from_wavelet_to_time,
)
from pywavelet.types import Wavelet


def test_timedomain_chirp_roundtrip(plot_dir, chirp_time):
    __run_timedomain_checks(chirp_time, "roundtrip_chirp_time", plot_dir)


def test_timedomain_sine_roundtrip(plot_dir, sine_time):
    __run_timedomain_checks(sine_time, "roundtrip_sine_time", plot_dir)


def test_freqdomain_chirp_roundtrip(plot_dir, chirp_freq):
    __run_freqdomain_checks(chirp_freq, "roundtrip_chirp_freq", plot_dir)


def test_freqdomain_sine_roundtrip(plot_dir, sine_freq):
    __run_freqdomain_checks(sine_freq, "roundtrip_sine_freq", plot_dir)


def test_freqdomain_pure_f0_transform(plot_dir):
    Nf, Nt, dt = 8, 4, 0.1
    hf = generate_pure_f0(Nf=Nf, Nt=Nt, dt=dt)
    hf_1 = __run_freqdomain_checks(
        hf, "roundtrip_pure_f0_freq", plot_dir, Nf=Nf, dt=dt
    )
    plot_fft(hf, hf_1, f"{plot_dir}/test_pure_f0_transform.png")


def test_conversion_from_hf_ht():
    Nf, Nt, dt = 8, 4, 0.1
    hf = generate_pure_f0(Nf=Nf, Nt=Nt, dt=dt)
    ht = hf.to_timeseries()
    w1 = hf.to_wavelet(Nf=Nf, Nt=Nt)
    w2 = ht.to_wavelet(Nf=Nf, Nt=Nt)
    assert w1 == w2
    hf1 = w1.to_frequencyseries()
    ht2 = w2.to_timeseries()
    assert hf == hf1
    assert ht == ht2


def __run_freqdomain_checks(hf, label, outdir, Nf=Nf, dt=dt):
    wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    __assert_wavelet_matches_cached_wavelet(wavelet, label, outdir)
    h_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)
    plot_freqdomain_comparisions(
        hf, h_reconstructed, wavelet, f"{outdir}/{label}.png"
    )
    __assert_roundtrip_valid(hf, h_reconstructed, wavelet)
    return h_reconstructed


def __run_timedomain_checks(ht, label, outdir, Nt=Nt, mult=mult, dt=dt):
    wavelet = from_time_to_wavelet(ht, Nt=Nt, mult=mult)
    __assert_wavelet_matches_cached_wavelet(wavelet, label, outdir)
    h_reconstructed = from_wavelet_to_time(wavelet, mult=mult, dt=dt)
    plot_timedomain_comparisons(
        ht, h_reconstructed, wavelet, f"{outdir}/{label}.png"
    )
    __assert_roundtrip_valid(ht, h_reconstructed, wavelet)


def __assert_roundtrip_valid(h_old, h_new, wavelet):
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


def __assert_wavelet_matches_cached_wavelet(cur: Wavelet, label, outdir):
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

    plot_wavelet_comparison(cur, cached, err, label, outdir)

    assert net_err < 1e-3, f"Net error is too large: {net_err}"
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
