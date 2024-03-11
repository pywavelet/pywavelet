import bilby
import matplotlib.pyplot as plt
import numpy as np
from gw_utils import DT, DURATION, get_ifo

from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import Wavelet

Nf, Nt = 1024, 1024
# Nf, Nt = 64, 64
ND = Nf * Nt
T_GRID = np.arange(0, ND) * DT
F_GRID = np.arange(0, ND // 2 + 1) * 1 / (DURATION)

t_binwidth = DURATION / Nt
f_binwidth = 1 / 2 * t_binwidth
fmax = 1 / (2 * DT)


T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, fmax, Nf)


def test_wavelet_psd_from_stationary(plot_dir):
    """n: number of noise wavelets to take median of"""
    ifo: bilby.gw.detector.Interferometer = get_ifo()[0]
    psd = ifo.power_spectral_density.psd_array
    psd_f = ifo.power_spectral_density.frequency_array

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

    plt.savefig(f"{plot_dir}/psd_wavelet.png", dpi=300)


    plt.close()
    # plot the normal PSD and the wavelet PSD (one time bin) in loglog
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.loglog(psd_f, psd, label="Normal PSD")
    plt.loglog(psd_wavelet.freq.data, np.exp(psd_wavelet.data[:, 0]) , label="Wavelet PSD")
    plt.show()