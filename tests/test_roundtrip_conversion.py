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
        Nt,
        mult,
        dt,
        f"{plot_dir}/out_roundtrip/chirp_time.png",
    )


def test_timedomain_sine_roundtrip(make_plots, plot_dir):
    f_true = 10
    __run_timedomain_checks(
        generate_sine_time_domain_signal(ts, ND, f_true=f_true),
        Nt,
        mult,
        dt,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine_time.png",
    )


# def test_freqdomain_chirp_roundtrip(make_plots, plot_dir):
#     freq_range = [20, 100]
#     ht = generate_chirp_time_domain_signal(ts, freq_range),
#
#     hf =
#
#     Nt = Nt,
#     mult = mult,
#     minimum_frequency = freq_range[0],
#     maximum_frequency = freq_range[1],
#
# ).frequencyseries
#     __run_freqdomain_checks(
#     hf,
#     Nt,
#     dt,
#     make_plots,
#     f"{plot_dir}/out_roundtrip/chirp_freq.png",
#     )

def test_freqdomain_sine_roundtrip(make_plots, plot_dir):
    f0 = 20
    T = 1000
    A = 2
    Nf = 32

    dt = 0.5 / (
            2 * f0
    )  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array

    # round len(t) to the nearest power of 2
    t = t[: 2 ** int(np.log2(len(t)))]
    T = len(t) * dt
    ND = len(t)

    h = A * np.sin(
        2 * np.pi * t * (f0 + 0 * t)
    )  # Signal waveform we wish to test

    Nt = ND // Nf

    frequencies = np.fft.rfftfreq(ND, d=dt)
    fft_data = np.fft.rfft(h)

    freqseries = FrequencySeries(data=fft_data, freq=frequencies)
    timeseries = TimeSeries(data=h, time=t)

    __run_freqdomain_checks(
        freqseries,
        Nf,
        dt,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine_freq.png",
    )

    __run_timedomain_checks(
        timeseries,
        Nt,
        mult,
        dt,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine_time.png",
    )


def __run_freqdomain_checks(hf, Nf, dt, make_plots, fname):
    h_wavelet = CoupledData.from_frequencyseries(hf, Nf=Nf)
    h_reconstructed = from_wavelet_to_freq(h_wavelet.wavelet, dt=dt)

    if make_plots:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        __make_plots(
            hf,
            h_reconstructed,
            h_wavelet,
            fname,
        )
    __check_residuals(hf.data - h_reconstructed.data, "t->f->wdm->f")


def __run_timedomain_checks(ht, Nt, mult, dt, fname):
    wavelet = from_time_to_wavelet(ht, Nt=Nt, mult=mult)
    assert wavelet.shape == (Nt, Nf)
    assert wavelet.__repr__() == f"Wavelet(NtxNf={Nt}x{Nf})"
    assert len(wavelet.freq) == Nf
    assert len(wavelet.time) == Nt
    h_reconstructed = from_wavelet_to_time(wavelet, mult=mult, dt=dt)
    _make_time_domain_plots(ht, h_reconstructed, wavelet, fname)


def _make_time_domain_plots(ht, h_reconstructed, wavelet, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ht.plot(ax=axes[0], label="Original")
    h_reconstructed.plot(ax=axes[0], label="Reconstructed", linestyle="--", color="tab:orange")
    axes[0].legend()
    wavelet.plot(ax=axes[1])

    # assert no nans
    assert not np.isnan(h_reconstructed.data).any(), "Reconstructed data contains NaNs"

    plot_residuals(ht.data - h_reconstructed.data, axes[2])
    plt.tight_layout()
    plt.savefig(fname)
