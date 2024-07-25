import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    plot_residuals,
)

from pywavelet.data import Data, FrequencySeries, TimeSeries
from pywavelet.transforms import (
    from_time_to_wavelet,
    from_wavelet_to_freq,
    from_wavelet_to_time,
)

fs = 512
dt = 1 / fs
Nt = 2**6
Nf = 2**6
mult = 16
ND = Nt * Nf
ts = np.arange(0, ND) * dt


def test_timedomain_chirp_roundtrip(make_plots, plot_dir):
    freq_range = [20, 100]
    __run_timedomain_checks(
        generate_chirp_time_domain_signal(ts, freq_range),
        Nt,
        mult,
        dt,
        freq_range,
        make_plots,
        f"{plot_dir}/out_roundtrip/chirp_time.png",
    )


def test_timedomain_sine_roundtrip(make_plots, plot_dir):
    f_true = 10
    __run_timedomain_checks(
        generate_sine_time_domain_signal(ts, ND, f_true=f_true),
        Nt,
        mult,
        dt,
        [f_true - 5, f_true + 5],
        make_plots,
        f"{plot_dir}/out_roundtrip/sine_time.png",
    )


def test_freqdomain_chirp_roundtrip(make_plots, plot_dir):
    freq_range = [20, 100]
    hf = Data.from_timeseries(
        generate_chirp_time_domain_signal(ts, freq_range),
        Nt=Nt,
        mult=mult,
        minimum_frequency=freq_range[0],
        maximum_frequency=freq_range[1],
    ).frequencyseries
    __run_freqdomain_checks(
        hf,
        Nt,
        mult,
        dt,
        make_plots,
        f"{plot_dir}/out_roundtrip/chirp_freq.png",
    )


def test_freqdomain_sine_roundtrip(make_plots, plot_dir):
    f_true = 10
    # freqseries = generate_sine_time_domain_signal(ts, ND, f_true=f_true)
    # hf = Data.from_timeseries(freqseries, Nt=Nt, mult=mult).frequencyseries

    # Create the frequency array (one-sided)
    ND = 32 * 32
    frequencies = np.fft.fftshift(np.fft.fftfreq(ND, d=dt))
    frequencies = frequencies[ND // 2 :]
    frequencies = np.append(frequencies, fs)
    freq_index = np.where(np.isclose(frequencies, f_true))[0][0]
    fft_data = np.zeros(ND + 1, dtype=complex)
    fft_data[freq_index] = 1

    ## THIS MAKES THE SIGNAL --> ND+1 IN LEN

    freqseries = FrequencySeries(data=fft_data, freq=frequencies)

    __run_freqdomain_checks(
        freqseries,
        32,
        mult,
        dt,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine_freq.png",
    )


def test_freqdomain_roundtrip(make_plots, plot_dir):
    pass


def __run_timedomain_checks(ht, Nt, mult, dt, freq_range, make_plots, fname):
    data = Data.from_timeseries(
        ht,
        Nt=Nt,
        mult=mult,
        minimum_frequency=freq_range[0],
        maximum_frequency=freq_range[1],
    )
    h_reconstructed = from_wavelet_to_time(data.wavelet, mult=mult, dt=dt)
    if make_plots:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        __make_plots(
            ht,
            h_reconstructed,
            data,
            fname=fname,
        )
    __check_residuals(ht.data - h_reconstructed.data, "t->wdm->t")


def __run_freqdomain_checks(hf, Nt, mult, dt, make_plots, fname):
    data = Data.from_frequencyseries(hf, Nt=Nt, mult=mult)
    h_reconstructed = from_wavelet_to_freq(data.wavelet, dt=dt)

    if make_plots:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        __make_plots(
            hf,
            h_reconstructed,
            data,
            fname,
        )
    __check_residuals(hf.data - h_reconstructed.data, "t->f->wdm->f")


def __make_plots(h, h_reconstructed, data, fname):
    fig, axes = plt.subplots(6, 1, figsize=(4, 15))
    data.plot_all(axes=axes)
    plot_residuals(h.data - h_reconstructed.data, axes=axes[4:])
    plt.tight_layout()
    fig.savefig(fname, dpi=300)


def __check_residuals(residuals, label):
    mu, sig = np.mean(residuals), np.std(residuals)
    assert np.abs(mu) < 0.1, f"Roundtrip [{label}] residual mean is {mu} > 0.1"
    assert np.abs(sig) < 1, f"Roundtrip [{label}] residual std is {sig} > 1"
