import numpy as np
from jax import numpy as jnp
from typing import List
from scipy.signal import chirp
from pywavelet.transforms.phi_computer import phitilde_vec_norm
from numpy import fft
from pywavelet.transforms.to_wavelets import from_freq_to_wavelet
from pywavelet.transforms.from_wavelets import from_wavelet_to_freq
from pywavelet.transforms.types import FrequencySeries
import matplotlib.pyplot as plt

FREQ_RANGE = (10, 100)


def simulate_data(Nf):
    # assert Nf is power of 2
    assert Nf & (Nf - 1) == 0, "Nf must be a power of 2"
    fs = 512
    dt = 1 / fs
    Nt = Nf
    mult = 16
    nx = 4.0
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    y = chirp(t, f0=FREQ_RANGE[0], f1=FREQ_RANGE[1], t1=t[-1], method="quadratic")
    phif = phitilde_vec_norm(Nf, Nt, dt=dt, d=nx)
    yf = fft.fft(y)[:ND // 2 ]
    # only keep the positive frequencies
    freq = fft.fftfreq(ND, dt)[:ND // 2 ]
    freqseries = FrequencySeries(data=jnp.array(yf), freq=jnp.array(freq))
    freqseries.plot_periodogram()
    return freqseries, jnp.array(phif)


def test_freq_to_wdm(plot_dir):
    Nf = Nt = 512
    yf, phif = simulate_data(Nf)
    wdm = from_freq_to_wavelet(yf, Nf, Nt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    wdm.plot(ax=axes[0])
    yf.plot_periodogram(ax=axes[1])

    new_yf = from_wavelet_to_freq(wdm, dt=yf.dt)
    new_yf.plot_periodogram(ax=axes[1], zorder=-1)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/freq_to_wdm.png")


    error = yf.data - new_yf.data
    plt.hist(error, bins=100)
    plt.savefig(f"{plot_dir}/freq_to_wdm_error.png")



