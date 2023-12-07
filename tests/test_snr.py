from typing import Tuple

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gw_utils import DT, DURATION, get_ifo, inject_signal_in_noise
from matplotlib import colors

from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.utils.snr import compute_snr

Nf, Nt = 64, 64
ND = Nf * Nt
T_BINWIDTH = DURATION / Nt
F_BINWIDTH = 1 / 2 * T_BINWIDTH
FMAX = 1 / (2 * DT)

T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, FMAX, Nf)


def get_noise_wavelet_data(t0: float) -> TimeSeries:
    noise = get_ifo(t0)[0].strain_data.time_domain_strain
    noise_wavelet = from_time_to_wavelet(noise, Nf, Nt)
    return noise_wavelet


def get_wavelet_psd_from_median_noise() -> Wavelet:
    """n: number of noise wavelets to take median of"""
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    return evolutionary_psd_from_stationary_psd(
        psd=ifo.power_spectral_density.psd_array,
        psd_f=ifo.power_spectral_density.frequency_array,
        f_grid=F_GRID,
        t_grid=T_GRID,
    )


@pytest.mark.parametrize("distance", [10, 1000])
def test_snr(plot_dir, distance):
    h, timeseries_snr = inject_signal_in_noise(mc=30, q=1, distance=distance)
    htrue, _ = inject_signal_in_noise(
        mc=30, q=1, distance=distance, noise=False
    )

    data_wavelet = from_time_to_wavelet(h, Nt=Nt)
    h_wavelet = from_time_to_wavelet(htrue, Nt=Nt)
    psd_wavelet = get_wavelet_psd_from_median_noise()
    wavelet_snr = compute_snr(h_wavelet, data_wavelet, psd_wavelet)

    h = h_wavelet
    d = data_wavelet
    psd = psd_wavelet

    h_hat = h / np.sqrt(np.tensordot(h.T, h))
    psd = psd.assign_coords(time=d.time)
    d_hat = d / psd

    # plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    h_wavelet.plot(ax=ax[0, 0])
    ax[0, 0].set_title("h_wavelet")
    data_wavelet.plot(ax=ax[0, 1])
    ax[0, 1].set_title("data_wavelet")
    psd_wavelet.plot(ax=ax[0, 2])
    ax[0, 2].set_title("psd_wavelet")
    h_hat.plot(ax=ax[1, 0])
    ax[1, 0].set_title("h_hat")
    d_hat.plot(ax=ax[1, 1])
    ax[1, 1].set_title("d/PSD")
    (h_hat * d_hat).plot(ax=ax[1, 2])
    ax[1, 2].set_title("h_hat * d/PSD")
    plt.savefig(f"{plot_dir}/snr_computation_d{distance}.png", dpi=300)

    assert isinstance(wavelet_snr, float)
    assert wavelet_snr == timeseries_snr
