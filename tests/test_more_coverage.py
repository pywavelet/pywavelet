import importlib
import os

import numpy as np
import pytest


def test_wavelet_bins_preprocess_typeerror():
    from pywavelet.types.wavelet_bins import _preprocess_bins

    with pytest.raises(TypeError):
        _preprocess_bins(object(), Nf=4, Nt=4)


@pytest.mark.parametrize(
    "module_path",
    [
        "pywavelet.transforms.numpy.forward.main",
        "pywavelet.transforms.jax.forward.main",
    ],
)
def test_forward_time_to_wavelet_truncates_and_caps_mult(module_path, caplog):
    mod = importlib.import_module(module_path)

    from pywavelet.types import TimeSeries

    # Pick explicit Nf/Nt so ND is known; make the input longer so truncation is safe.
    Nf, Nt = 4, 4
    ND = Nf * Nt
    dt = 0.1
    time = np.arange(0, ND + 4) * dt
    data = np.linspace(0, 1, len(time))
    ts = TimeSeries(data=data, time=time)

    caplog.set_level("WARNING")
    w = mod.from_time_to_wavelet(ts, Nf=Nf, Nt=Nt, mult=10)

    assert w.Nf == Nf
    assert w.Nt == Nt
    assert any("Truncating" in rec.message for rec in caplog.records)
    assert any("mult=" in rec.message for rec in caplog.records)


def test_frequencyseries_validations_and_roundtrip_time_conversion():
    from pywavelet.types import FrequencySeries

    # validation paths
    with pytest.raises(ValueError):
        FrequencySeries(data=np.zeros(2), freq=np.array([-1.0, 0.0]))
    with pytest.raises(ValueError):
        FrequencySeries(data=np.zeros(2), freq=np.zeros(3))

    # to_timeseries preserves ND and t0 shift
    ND = 16
    dt = 0.25
    freq = np.fft.rfftfreq(ND, d=dt)
    data = np.zeros(len(freq), dtype=np.complex128)
    data[1] = 1.0 + 0.0j
    t0 = 3.0
    hf = FrequencySeries(data=data, freq=freq, t0=t0)
    ht = hf.to_timeseries()
    assert ht.ND == ND
    assert ht.t0 == pytest.approx(t0)

    # exercise inner-product methods
    psd = FrequencySeries(data=np.ones_like(data), freq=freq)
    ip = hf.noise_weighted_inner_product(hf, psd)
    assert np.isfinite(ip)
    snr = hf.matched_filter_snr(hf, psd)
    assert snr == pytest.approx(np.sqrt(ip))


def test_timeseries_padding_filtering_slicing_and_truncation(caplog):
    from pywavelet.types import TimeSeries

    dt = 0.1
    time = np.arange(0, 10) * dt
    data = np.ones_like(time)
    ts = TimeSeries(data=data, time=time)

    # zero_pad_to_power_of_2 warning branch (n_pad > 0)
    caplog.set_level("WARNING")
    padded = ts.zero_pad_to_power_of_2(tukey_window_alpha=0.1)
    assert padded.ND == 16
    assert any("Padding the data" in rec.message for rec in caplog.records)

    # zero_pad_to_power_of_2 no-warning branch (already pow2)
    caplog.clear()
    ts2 = TimeSeries(data=np.ones(8), time=np.arange(0, 8) * dt)
    padded2 = ts2.zero_pad_to_power_of_2()
    assert padded2.ND == 8
    assert not any("Padding the data" in rec.message for rec in caplog.records)

    # highpass_filter error paths and success path
    with pytest.raises(ValueError):
        ts.highpass_filter(fmin=0.0)
    with pytest.raises(ValueError):
        ts.highpass_filter(fmin=ts.nyquist_frequency + 1.0)
    # filtfilt-style padding needs a longer series than our tiny padding test
    ts_long = TimeSeries(data=np.ones(128), time=np.arange(0, 128) * dt)
    filtered = ts_long.highpass_filter(fmin=1.0)
    assert filtered.ND == ts_long.ND

    # slicing paths
    sl = ts[:5]
    assert sl.ND == 5
    one = ts[0]
    assert float(one) == 1.0

    # truncate
    tr = ts.truncate(tmin=0.2, tmax=0.6)
    assert tr.t0 >= 0.2 - 1e-12
    assert tr.tend <= 0.6 + tr.dt + 1e-12


def test_wavelet_properties_ops_and_snr():
    from pywavelet.types import Wavelet

    Nf, Nt = 4, 5
    dt = 0.2
    time = np.arange(Nt) * (Nf * dt)  # delta_T = Nf*dt, delta_t=dt
    freq = np.arange(Nf) * (1 / (2 * (Nf * dt)))
    data = np.ones((Nf, Nt), dtype=np.float64)
    w = Wavelet(data=data, time=time, freq=freq)

    assert w.delta_t == pytest.approx(dt)
    assert w.delta_T == pytest.approx(Nf * dt)
    assert w.delta_F == pytest.approx(1 / (2 * w.delta_T))
    assert w.delta_f == pytest.approx(1 / (2 * w.delta_t))
    assert w.fs == pytest.approx(1 / dt)
    assert w.nyquist_frequency == pytest.approx(0.5 / dt)

    # ops
    w2 = w / 2.0
    assert np.allclose(w2.data, 0.5)
    w3 = w / w
    assert np.allclose(w3.data, 1.0)
    w4 = w.copy()
    assert w4 == w

    # snr helpers
    psd = Wavelet(data=np.ones_like(w.data), time=w.time, freq=w.freq)
    expected = np.sqrt(w.noise_weighted_inner_product(w, psd))
    assert w.optimal_snr(psd) == pytest.approx(expected)


def test_plotting_smoke_and_time_axis_branches():
    import matplotlib.pyplot as plt

    from pywavelet.types.plotting import (
        _fmt_time_axis,
        plot_spectrogram,
        plot_wavelet_grid,
    )

    # plot_wavelet_grid with trend + log scale path
    Nf, Nt = 3, 7
    time = np.linspace(0, 1, Nt)
    freq = np.linspace(0, 10, Nf)
    w = np.abs(np.random.RandomState(0).randn(Nf, Nt)) + 1e-3
    w[0, 1] = np.nan
    w[1, 3] = np.nan

    fig1, ax1 = plot_wavelet_grid(
        w,
        time,
        freq,
        zscale="log",
        absolute=True,
        show_colorbar=True,
        trend_color="black",
        label="test",
    )
    plt.close(fig1)

    fig2, ax2 = plot_wavelet_grid(
        w,
        time,
        freq,
        zscale="linear",
        absolute=False,
        show_colorbar=False,
        detailed_axes=True,
        show_gridinfo=False,
    )
    plt.close(fig2)

    # _fmt_time_axis branches: seconds/min/hr/day
    fig, ax = plt.subplots()
    _fmt_time_axis(np.array([0.0, 10.0]), ax)
    _fmt_time_axis(np.array([0.0, 61.0]), ax)
    _fmt_time_axis(np.array([0.0, 3601.0]), ax)
    _fmt_time_axis(np.array([0.0, 90001.0]), ax)
    plt.close(fig)

    # plot_spectrogram smoke
    fig3, ax3 = plot_spectrogram(np.random.RandomState(1).randn(128), fs=64.0)
    plt.close(fig3)
