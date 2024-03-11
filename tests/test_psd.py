import bilby
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from gw_utils import DT, DURATION, get_ifo

from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_wavelet_to_time, from_time_to_wavelet
from pywavelet.transforms.types import Wavelet, FrequencySeries
from pywavelet.psd import generate_noise_from_psd

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


def _get_lvk_psd_func():
    psd, psd_f = _get_psd_freq_dom()
    return interp1d(psd_f, psd, bounds_error=False, fill_value=max(psd))


def test_wavelet_psd_from_stationary(plot_dir):
    """n: number of noise wavelets to take median of"""
    psd, psd_f = _get_psd_freq_dom()

     # Wavelet data from noise
    noise_ts = generate_noise_from_psd(
        psd_func=_get_lvk_psd_func(),
        n_data=2**17, fs=1 / DT,
    )
    noise_pdgrm = FrequencySeries.from_time_series(noise_ts, min_freq=min(psd_f), max_freq=max(psd_f))
    noise_wavelet  = from_time_to_wavelet(noise_ts, Nt=128, freq_range=(min(psd_f), max(psd_f)))


    # plot the noise-pdgrm and the true psd
    _ = noise_pdgrm.plot(color="tab:blue", lw=0, marker=",")
    plt.loglog(psd_f, psd, color="tab:orange", label="PSD", alpha=0.5, zorder=1)
    plt.show()

    # plot the noise-wavelet
    noise_wavelet.plot(absolute=True, zscale="log", freq_scale="linear", freq_range=(min(psd_f), None))
    plt.show()

    # generate and plot the true PSD --> wavelet
    psd_wavelet: Wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd,
        psd_f=psd_f,
        f_grid=noise_wavelet.freq.data,
        t_grid=noise_wavelet.time.data,
    )
    psd_wavelet.plot(
        absolute=True, zscale="log", freq_scale="linear", freq_range=(min(psd_f), max(F_GRID))
     )
    plt.show()

    # look at ratio of shit
    psd_an = np.sqrt(psd_wavelet.data * psd_wavelet.sample_rate /2 )
    ratio = (noise_wavelet.data / psd_an)
    plt.hist(ratio.flatten(), bins=100)
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
