import matplotlib.pyplot as plt
import numpy as np

from pywavelet.types import FrequencySeries, TimeSeries


def test_timefreq_type(sine_time):
    assert isinstance(sine_time, TimeSeries)
    sine_freq: FrequencySeries = sine_time.to_frequencyseries()
    assert len(sine_freq) == len(sine_time) // 2 + 1
    assert sine_freq.duration == sine_time.duration
    assert sine_freq.minimum_frequency == 0
    assert sine_freq.fs == sine_time.fs
    assert sine_freq.ND == sine_time.ND


def test_slicing(sine_time):
    N = len(sine_time)
    assert len(sine_time[0 : N // 2]) == N // 2
    assert len(sine_time[N // 2 :]) == N // 2
    assert len(sine_time[0 : N // 2 : 2]) == N // 4
    assert len(sine_time[N // 2 :: 2]) == N // 4


def test_zeropadding_and_filtering(plot_dir):
    N = 8000
    t = np.linspace(0, 10, N)
    dt = t[1] - t[0]
    fs = 1 / dt
    fmain = 10
    fmin = 1
    fcutoff = fmin + 0.5
    print(f"fs={fs}, fmin={fmin}")
    y = np.sin(2 * np.pi * fmain * t) + 10 * np.sin(2 * np.pi * fmin * t)

    fig, axes = plt.subplots(2, 1)
    label = r"$\sin(2\pi FMAIN t) + 10\sin(2\pi FMIN t)$"
    label = label.replace("FMAIN", f"{fmain}")
    label = label.replace("FMIN", f"{fmin}")
    fig.suptitle(label)
    ts = TimeSeries(y, t)
    ts_padded = ts.zero_pad_to_power_of_2(tukey_window_alpha=0.1)
    ts_filtered = ts.highpass_filter(fcutoff, tukey_window_alpha=0.1)

    fs = ts.to_frequencyseries()
    fs_padded = ts_padded.to_frequencyseries()
    fs_filtered = ts_filtered.to_frequencyseries()

    ts.plot(ax=axes[0])
    ts_padded.plot(ax=axes[0])
    ts_filtered.plot(ax=axes[0])
    fs.plot_periodogram(ax=axes[1], label="Original")
    fs_padded.plot_periodogram(ax=axes[1], label="Padded")
    fs_filtered.plot_periodogram(ax=axes[1], label="Filtered")
    axes[1].axvline(
        fcutoff, color="red", linestyle="--", label="Cutoff frequency"
    )
    axes[1].legend(loc="lower left")
    axes[1].set_ylim(bottom=min(fs_filtered.data**2))
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/test_zeropadding_and_filtering.png")
