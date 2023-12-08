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


def get_wavelet_psd_from_median_noise(f_grid=F_GRID, t_grid=T_GRID) -> Wavelet:
    """n: number of noise wavelets to take median of"""
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    return evolutionary_psd_from_stationary_psd(
        psd=ifo.power_spectral_density.psd_array,
        psd_f=ifo.power_spectral_density.frequency_array,
        f_grid=f_grid,
        t_grid=t_grid,
    )


@pytest.mark.parametrize("distance", [10, 100, 1000])
def test_snr(plot_dir, distance):
    data_time, timeseries_snr = inject_signal_in_noise(
        mc=30, q=1, distance=distance
    )
    h_time, _ = inject_signal_in_noise(
        mc=30, q=1, distance=distance, noise=False
    )

    data_wavelet = from_time_to_wavelet(data_time, Nt=Nt)
    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt)
    psd_wavelet = get_wavelet_psd_from_median_noise(
        f_grid=h_wavelet.freq.data, t_grid=h_wavelet.time.data
    )
    wavelet_snr = compute_snr(h_wavelet, data_wavelet, psd_wavelet)

    h = h_wavelet.data
    d = data_wavelet.data
    psd = psd_wavelet.data

    h_hat = h / np.sqrt(np.tensordot(h, h))
    d_hat = d / psd

    # plot WAVELET
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    h_wavelet.plot(ax=ax[0, 0])
    ax[0, 0].set_title("h_wavelet")
    data_wavelet.plot(ax=ax[0, 1])
    ax[0, 1].set_title("data_wavelet")
    cbar = ax[0, 2].imshow(
        np.log(np.rot90(psd.T)),
        aspect="auto",
        cmap="bwr",
        extent=[0, DURATION, 0, FMAX],
    )
    # add cbar to the right
    cbar = plt.colorbar(ax=ax[0, 2], mappable=cbar)
    ax[0, 2].set_title("log psd_wavelet")
    cbar = ax[1, 0].imshow(np.rot90(h_hat.T), aspect="auto", cmap="bwr")
    plt.colorbar(ax=ax[1, 0], mappable=cbar)
    cbar = ax[1, 1].imshow(np.rot90(d_hat.T), aspect="auto", cmap="bwr")
    plt.colorbar(ax=ax[1, 1], mappable=cbar)
    cbar = ax[1, 2].imshow(
        np.rot90((h_hat * d_hat).T), aspect="auto", cmap="bwr"
    )
    plt.colorbar(ax=ax[1, 2], mappable=cbar)
    ax[1, 0].set_title("h_hat")
    ax[1, 1].set_title("d/PSD")
    ax[1, 2].set_title("h_hat * d/PSD")
    plt.suptitle(
        f"Matched Filter SNR: {timeseries_snr:.2f}, Wavelet SNR: {wavelet_snr:.2E}"
    )
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/snr_computation_d{distance}.png", dpi=300)

    assert isinstance(wavelet_snr, float)
    assert wavelet_snr == timeseries_snr


def plot_spectograms(h_time, data_time):
    # plot normal scipy SPECTOGRAM
    plt.close("all")
    spec = plt.specgram(
        h_time, Fs=1 / DT, NFFT=256, mode="magnitude", scale_by_freq=False
    )
    plt.colorbar(spec[-1])
    plt.show()

    plt.close("all")
    spec = plt.specgram(
        data_time, Fs=1 / DT, NFFT=256, mode="magnitude", scale_by_freq=False
    )
    plt.colorbar(spec[-1])
    plt.show()

    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    psd = ifo.power_spectral_density.psd_array
    psd_f = ifo.power_spectral_density.frequency_array
    # FFT of data (get both frequencies and amplitudes)
    data_freq = np.fft.rfft(data_time)
    freq = np.fft.rfftfreq(len(data_time), DT)

    # filter data and psd to only have the same frequencies
    data_freq = data_freq[: len(psd_f)]
    freq = freq[: len(psd_f)]
    # get the psd at the frequencies of the data
    psd = np.interp(freq, psd_f, psd)
    # get the clean data

    clean_data = np.fft.irfft(data_freq / np.sqrt(psd))
    plt.close("all")
    spec = plt.specgram(
        clean_data, Fs=1 / DT, NFFT=256, mode="magnitude", scale_by_freq=False
    )
    plt.colorbar(spec[-1])
    plt.show()
