import numpy as np
import pytest


def test_timeseries_init_add_sub_eq_mul_copy():
    import matplotlib.pyplot as plt

    from pywavelet.types import TimeSeries

    dt = 0.1
    t = np.arange(0, 8) * dt
    a = TimeSeries(data=np.ones_like(t), time=t)
    b = TimeSeries(data=2 * np.ones_like(t), time=t)

    c = a + b
    assert isinstance(c, TimeSeries)
    assert np.allclose(c.data, 3.0)

    d = b - a
    assert np.allclose(d.data, 1.0)

    assert a == a.copy()
    assert (a * 2.0) == b

    # error branches: mismatched shapes
    t2 = np.arange(0, 7) * dt
    short = TimeSeries(data=np.ones_like(t2), time=t2)
    with pytest.raises(ValueError):
        _ = a + short
    with pytest.raises(ValueError):
        _ = a - short

    # init length mismatch
    with pytest.raises(ValueError):
        TimeSeries(data=np.ones(3), time=np.ones(4))

    # plot_spectrogram wrapper (smoke)
    fig, ax = a.plot_spectrogram()
    plt.close(fig)

    # _EMPTY helper
    empty = TimeSeries._EMPTY(ND=16, dt=dt)
    assert empty.ND == 16


def test_plotting_helpers_ax_none_and_existing_ax():
    import matplotlib.pyplot as plt

    from pywavelet.types.plotting import (
        plot_freqseries,
        plot_periodogram,
        plot_timeseries,
        plot_wavelet_grid,
        plot_wavelet_trend,
    )

    # freq/time plots: ax=None branch
    freq = np.linspace(0, 10, 11)
    data = np.linspace(0, 1, 11)
    fig1, ax1 = plot_freqseries(data, freq, nyquist_frequency=10.0)
    plt.close(fig1)

    fig2, ax2 = plot_periodogram(data.astype(np.complex128), freq, 10.0)
    plt.close(fig2)

    time = np.linspace(0, 1, 11)
    fig3, ax3 = plot_timeseries(data, time)
    plt.close(fig3)

    # ax provided path
    fig4, ax4 = plt.subplots()
    plot_freqseries(data, freq, nyquist_frequency=10.0, ax=ax4)
    plt.close(fig4)

    # plot_wavelet_trend: ax=None branch
    w = np.random.RandomState(0).randn(4, 12)
    tg = np.linspace(0, 1, 12)
    fg = np.linspace(0, 10, 4)
    plot_wavelet_trend(w, tg, fg, ax=None)
    plt.close(plt.gcf())

    # plot_wavelet_grid shape mismatch error branch
    with pytest.raises(ValueError):
        plot_wavelet_grid(
            wavelet_data=np.zeros((3, 3)),
            time_grid=np.zeros(4),
            freq_grid=np.zeros(3),
        )


def test_wavelet_add_sub_mask_and_matched_filter_snr():
    from pywavelet.types import Wavelet, WaveletMask

    Nf, Nt = 4, 6
    time = np.linspace(0, 1, Nt)
    freq = np.linspace(0, 20, Nf)
    w1 = Wavelet(data=np.ones((Nf, Nt)), time=time, freq=freq)
    w2 = Wavelet(data=2 * np.ones((Nf, Nt)), time=time, freq=freq)

    assert np.allclose((w1 + w2).data, 3.0)
    assert np.allclose((w1 + 1.0).data, 2.0)
    assert np.allclose((w2 - 1.0).data, 1.0)

    # Mask multiplication path (sets masked regions to NaN)
    mask = WaveletMask.from_restrictions(
        time_grid=time, freq_grid=freq, frange=[5.0, 15.0], tgaps=[(0.3, 0.7)]
    )
    masked = w1 * mask
    assert np.isnan(masked.data).any()
    assert "WaveletMask" in repr(mask)

    # matched filter SNR path
    psd = Wavelet(data=np.ones((Nf, Nt)), time=time, freq=freq)
    snr = w1.matched_filter_snr(template=w1, psd=psd)
    assert np.isfinite(float(snr))

    # NotImplemented paths (unsupported operand type)
    assert Wavelet.__add__(w1, object()) is NotImplemented
    assert Wavelet.__sub__(w1, object()) is NotImplemented
    assert Wavelet.__mul__(w1, object()) is NotImplemented
    assert Wavelet.__truediv__(w1, object()) is NotImplemented
