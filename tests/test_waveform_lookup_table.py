from pywavelet.waveform_generator.generators import FunctionalWaveformGenerator, LookupTableWaveformGenerator
import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    cbc_waveform,
    plot_wavelet_domain_signal
)
import matplotlib.pyplot as plt


def test_waveform_function(plot_dir):
    fig = plt.figure()
    for i, mc in enumerate(range(10, 50, 5)):
        t, h = cbc_waveform(mc)
        h = h / h.max()
        plt.plot(t, h + i, label=f"mc={mc}")
    plt.legend()
    plt.savefig(f"{plot_dir}/cbc_waveforms.png", dpi=300)


def test_waveform_lookup_table(plot_dir):
    ND = 2048
    dt = 1 / 256
    fmin = 20
    h_func = lambda mc: cbc_waveform(mc, q=1, delta_t=dt, f_lower=fmin)[1]
    t = cbc_waveform(15, q=1, delta_t=dt, f_lower=fmin)[0]
    t = t - t.min()
    dt = t[1] - t[0]
    Tobs = max(t)
    fs = np.arange(0, ND // 2 + 1) * 1 / Tobs

    Nf, Nt = 64, 64
    mult = 16

    plt.plot(t, h_func(15))
    plt.show()
    waveform_generator = FunctionalWaveformGenerator(h_func, Nf=Nf, Nt=Nt, mult=mult)



    # time and frequency grids

    for i, mc in enumerate(range(15, 50, 5)):
        wavelet_matrix = waveform_generator(mc=mc)
        fig = plot_wavelet_domain_signal(wavelet_matrix, t, fs, (0, 64))
        fig.suptitle(f"mc={mc}")
        fig.show()


