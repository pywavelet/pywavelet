import bilby
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors

from typing import Tuple
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.utils.snr import compute_snr

from gw_utils import DT, DURATION, get_ifo, inject_signal_in_noise

Nf, Nt = 64, 64
ND = Nf * Nt
T_BINWIDTH = DURATION / Nt
F_BINWIDTH = 1 / 2 * T_BINWIDTH
FMAX = 1 / (2 * DT)

T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, FMAX, Nf)

def get_noise_wavelet_data(t0:float)->TimeSeries:
    noise = get_ifo(t0)[0].strain_data.time_domain_strain
    noise_wavelet = from_time_to_wavelet(noise, Nf, Nt)
    return noise_wavelet


def get_wavelet_psd_from_median_noise()->Wavelet:
    """n: number of noise wavelets to take median of"""
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    return evolutionary_psd_from_stationary_psd(
        psd=ifo.power_spectral_density.psd_array,
        psd_f=ifo.power_spectral_density.frequency_array,
        f_grid=F_GRID,
        t_grid=T_GRID,
    )


@pytest.mark.parametrize("distance", [10])
def test_snr(distance):
    h, timeseries_snr = inject_signal_in_noise(mc=30, q=1, distance=distance)

    data_wavelet = from_time_to_wavelet(h, Nt=Nt)
    data_wavelet.plot()
    plt.savefig("data_wavelet.png", dpi=300)

    h_wavelet = from_time_to_wavelet(h, Nt=Nt)
    h_wavelet.plot()
    plt.savefig("h_wavelet.png", dpi=300)

    psd_wavelet = get_wavelet_psd_from_median_noise()
    wavelet_snr = compute_snr(h_wavelet, data_wavelet, psd_wavelet)
    assert isinstance(wavelet_snr, float)
    assert wavelet_snr == timeseries_snr
