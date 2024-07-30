import os

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    plot_residuals,
)

from pywavelet.data import CoupledData, TimeSeries

dt = 1 / 512
Nt = 64
Nf = 128
mult = 16
ND = Nt * Nf
ts = np.arange(0, ND) * dt


def test_ts_plots(plot_dir):
    plt_dir = f"{plot_dir}/out_types"
    os.makedirs(plt_dir, exist_ok=True)
    frange = [20, 50]
    htime = generate_chirp_time_domain_signal(t=ts, freq_range=frange)
    data_chirp = CoupledData.from_timeseries(
        htime,
        minimum_frequency=frange[0],
        maximum_frequency=frange[1],
        Nt=Nt,
        Nf=Nf,
        mult=mult,
    )
    hsine = generate_sine_time_domain_signal(ts, ND, f_true=25)
    data_sine = CoupledData.from_timeseries(
        hsine,
        minimum_frequency=frange[0],
        maximum_frequency=frange[1],
        Nt=Nt,
        Nf=Nf,
        mult=mult,
    )

    __roundtrip(data_chirp, f"{plt_dir}/chirp_series.png")
    __roundtrip(data_sine, f"{plt_dir}/sine_series.png")


def __roundtrip(data: CoupledData, fname: str, **kwargs):
    assert data.frequencyseries.duration == data.timeseries.duration
    data_reconstructed = CoupledData.from_frequencyseries(
        data.frequencyseries,
        start_time=data.start_time,
        Nt=data.Nt,
        Nf=data.Nf,
        mult=data.mult,
    )
    residuals = data.timeseries - data_reconstructed.timeseries
    fig, axes = plt.subplots(5, 1, figsize=(4, 15))

    # plot time domain signal
    data.plot_all(axes=axes)
    plot_residuals(residuals, ax=axes[-1])
    fig.savefig(fname, dpi=300)

    residuals_mean = np.abs(residuals.data).mean()
    residuals_std = np.abs(residuals.data).std()
    assert residuals_mean < 0.1
    assert residuals_std < 1
