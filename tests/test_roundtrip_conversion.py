import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    plot_residuals,
    plot_time_domain_signal,
    plot_wavelet_domain_signal,
)

from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time


def test_time_to_wavelet_to_time(make_plots, plot_dir):
    dt = 1 / 512
    Nt = 2**6
    Nf = 2**6
    mult = 16

    freq_range = (10, 0.2 * (1 / dt))

    ND = Nt * Nf

    # time and frequency grids
    ts = np.arange(0, ND) * dt
    Tobs = max(ts)
    fs = np.arange(0, ND // 2 + 1) * 1 / (Tobs)

    # generate signal
    h_time = generate_chirp_time_domain_signal(ts, freq_range)
    h_wavelet = from_time_to_wavelet(h_time, Nf=Nf, Nt=Nt, mult=mult)
    h_reconstructed = from_wavelet_to_time(
        h_wavelet.data, Nf=Nf, Nt=Nt, mult=mult
    )

    if make_plots:
        fig = plot_time_domain_signal(ts, h_time, freq_range)
        fig.savefig(f"{plot_dir}/original_signal.png", dpi=300)

        fig = plot_wavelet_domain_signal(h_wavelet.data, ts, fs, freq_range)
        fig.savefig(f"{plot_dir}/wavelet_domain.png", dpi=300)

        fig = plot_time_domain_signal(ts, h_reconstructed, freq_range)
        fig.savefig(f"{plot_dir}/reconstructed_signal.png", dpi=300)

        fig = plot_residuals(h_time - h_reconstructed)
        fig.savefig(f"{plot_dir}/residuals.png", dpi=300)

    # check that the reconstructed signal is the same as the original
    assert np.allclose(h_time, h_reconstructed, atol=1e-3)
