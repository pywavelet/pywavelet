import bilby
import matplotlib.pyplot as plt
import numpy as np
import scipy
from gw_utils import DT, DURATION, get_ifo

from pywavelet.fft_funcs import periodogram
from pywavelet.psd import (
    _generate_noise_from_psd,
    evolutionary_psd_from_stationary_psd,
    get_noise_wavelet_from_psd,
)
from pywavelet.transforms import from_wavelet_to_time
from pywavelet.transforms.types import Wavelet

Nf, Nt = 1024, 1024
# Nf, Nt = 64, 64
ND = Nf * Nt
T_GRID = np.arange(0, ND) * DT
F_GRID = np.arange(0, ND // 2 + 1) * 1 / (DURATION)
F_SAMP = 1 / DT

t_binwidth = DURATION / Nt
f_binwidth = 1 / 2 * t_binwidth
fmax = 1 / (2 * DT)


T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, fmax, Nf)


def _get_psd_freq_dom():
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    psd = ifo.power_spectral_density.psd_array
    psd_f = ifo.power_spectral_density.frequency_array
    return psd, psd_f


def test_wavelet_psd_from_stationary(plot_dir):
    """n: number of noise wavelets to take median of"""
    psd, psd_f = _get_psd_freq_dom()

    psd_wavelet: Wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd,
        psd_f=psd_f,
        f_grid=F_GRID,
        t_grid=T_GRID,
    )
    psd_wavelet.data = np.log(psd_wavelet.data)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(np.log(psd.T), psd_f)
    axes[0].set_ylim(0, 256)
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_xlabel("log PSD")
    psd_wavelet.plot(ax=axes[1], cmap=None)
    # change colorbar label
    fig.get_axes()[-1].set_ylabel("Log Wavelet Amplitude")

    plt.savefig(f"{plot_dir}/psd_wavelet.png", dpi=300)

    psd_wavelet.data = np.nan_to_num(
        psd_wavelet.data, nan=np.max(psd_wavelet.data)
    )
    noise_ts = from_wavelet_to_time(psd_wavelet)
    # generate welch_psd
    freq, welch_psd = scipy.signal.welch(noise_ts, fs=1 / DT, nperseg=1024)
    plt.figure()
    plt.loglog(freq, welch_psd, color="tab:blue", label="Welch PSD")
    plt.loglog(psd_f, psd, color="tab:orange", label="PSD", alpha=0.5, ls="--")
    plt.show()


def test_bahgi_psd_technique(plot_dir):
    # METHOD 1: load S(f) --> generate timeseries --> wavelet transform
    psd, psd_f = _get_psd_freq_dom()
    noise_ts = _generate_noise_from_psd(psd, psd_f, DURATION * 2048, F_SAMP)

    # generate periodogram
    noise_pdgmr = periodogram(noise_ts)
    plt.figure()
    plt.loglog(noise_pdgmr.freq, noise_pdgmr.data)
    plt.loglog(psd_f, psd)
    plt.show()

    noise_wavelet = get_noise_wavelet_from_psd(
        duration=DURATION * 2048,
        sampling_freq=1 / DT,
        psd_f=psd_f,
        psd=psd,
        Nf=Nf,
    )

    noise_wavelet.plot()
    plt.savefig(f"{plot_dir}/bahgi_psd_technique.png", dpi=300)
    # replace nans with zeros
    noise_wavelet.data = np.nan_to_num(noise_wavelet.data)

    # the following doest work as we have nans in the noise-wavelet (at the low freq)
    noise_ts = from_wavelet_to_time(noise_wavelet)
    # generate periodogram
    freq, welch_psd = scipy.signal.welch(noise_ts, fs=1 / DT, nperseg=1024)
    plt.figure()
    plt.loglog(freq, welch_psd)
    plt.loglog(psd_f, psd)
    plt.show()
