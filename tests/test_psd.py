from gw_utils import DURATION, get_ifo, DT
from pywavelet.psd import evolutionary_psd_from_stationary_psd
import numpy as np
import bilby
from pywavelet.transforms.types import Wavelet
import matplotlib.pyplot as plt

Nf, Nt = 1024, 1024
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

    psd_wavelet:Wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd,
        psd_f=psd_f,
        f_grid=F_GRID,
        t_grid = T_GRID,
    )
    psd_wavelet.plot(cmap=None)
    plt.savefig(f"{plot_dir}/psd_wavelet.png", dpi=300)
