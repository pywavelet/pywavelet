import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import tukey
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    plot_residuals,
)

from pywavelet.data import Data, TimeSeries
from pywavelet.transforms import (
    from_time_to_wavelet,
    from_wavelet_to_freq,
    from_wavelet_to_time,
)

dt = 1 / 512
Nt = 2**6
Nf = 2**6
mult = 16
ND = Nt * Nf
ts = np.arange(0, ND) * dt


def test_chirp_roundtrip(make_plots, plot_dir):
    freq_range = [20, 100]
    h_time = generate_chirp_time_domain_signal(ts, freq_range)
    __run_checks(
        h_time,
        Nt,
        mult,
        dt,
        freq_range,
        make_plots,
        f"{plot_dir}/out_roundtrip/chirp.png",
    )


def test_sine_roundtrip(make_plots, plot_dir):
    f_true = 10
    freq_range = [f_true - 5, f_true + 5]
    h_time = generate_sine_time_domain_signal(ts, ND, f_true=f_true)
    __run_checks(
        h_time,
        Nt,
        mult,
        dt,
        freq_range,
        make_plots,
        f"{plot_dir}/out_roundtrip/sine.png",
    )


def __run_checks(h_time, Nt, mult, dt, freq_range, make_plots, fname):
    data_from_t = Data.from_timeseries(
        h_time,
        Nt=Nt,
        mult=mult,
        minimum_frequency=freq_range[0],
        maximum_frequency=freq_range[1],
    )
    h_freq = data_from_t.frequencyseries
    data_from_f = Data.from_frequencyseries(h_freq, Nt=Nt, mult=mult)
    h_reconstructed_from_time = from_wavelet_to_time(
        data_from_t.wavelet, mult=mult, dt=dt
    )
    h_reconstructed_from_freq = from_wavelet_to_freq(
        data_from_f.wavelet, dt=dt
    )

    if make_plots:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        __make_plots(
            h_time,
            h_reconstructed_from_time,
            data_from_t,
            fname=fname.replace(".png", "_time.png"),
        )
        __make_plots(
            h_freq,
            h_reconstructed_from_freq,
            data_from_f,
            fname=fname.replace(".png", "_freq.png"),
        )

    residuals_f = (
        data_from_t.frequencyseries.data - h_reconstructed_from_freq.data
    )
    f_mean, f_std = residuals_f.mean(), residuals_f.std()
    assert (
        np.abs(f_mean) < 0.1
    ), f"Roundtrip [f->wdm->t] residual mean is {f_std} > 0.1"
    assert (
        np.abs(f_std) < 1
    ), f"Roundtrip [f->wdm->t] residual std is {f_mean} > 1"

    residuals_t = h_time.data - h_reconstructed_from_time.data
    t_mean, t_std = residuals_t.mean(), residuals_t.std()
    assert (
        np.abs(t_mean) < 0.1
    ), f"Roundtrip [t->wdm->t] residual mean is {t_mean} > 0.1"
    assert (
        np.abs(t_std) < 1
    ), f"Roundtrip [t->wdm->t] residual std is {t_std} > 1"


def __make_plots(h, h_reconstructed, data, fname):
    fig, axes = plt.subplots(6, 1, figsize=(4, 15))
    data.plot_all(axes=axes)
    plot_residuals(h.data - h_reconstructed.data, axes=axes[4:])
    plt.tight_layout()
    fig.savefig(fname, dpi=300)
