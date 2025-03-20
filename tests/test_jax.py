import importlib
import os

import matplotlib.pyplot as plt
import numpy as np

from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd


def test_toy_model_snr(plot_dir):
    f0 = 20
    dt = 0.0125
    A = 2
    Nt = 128
    Nf = 256
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    PSD_AMP = 1

    ########################################
    # Part1: Analytical SNR calculation
    #######################################

    # Eq 21
    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test
    signal_timeseries = TimeSeries(y, t)
    signal_freq = signal_timeseries.to_frequencyseries()
    psd_freq = FrequencySeries(
        PSD_AMP * np.ones(len(signal_freq)), signal_freq.freq
    )
    snr = signal_freq.optimal_snr(psd_freq)

    ########################################
    # Part2: Wavelet domain (numpy)
    ########################################

    signal_wavelet = from_freq_to_wavelet(signal_freq, Nf=Nf, Nt=Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd_freq.data,
        psd_f=psd_freq.freq,
        f_grid=signal_wavelet.freq,
        t_grid=signal_wavelet.time,
        dt=dt,
    )
    wdm_snr = compute_snr(signal_wavelet, signal_wavelet, psd_wavelet)
    assert np.isclose(snr, wdm_snr, atol=0.5), f"{snr}!={wdm_snr}"

    ########################################
    # Part3: Wavelet domain (jax)
    ########################################

    from pywavelet.transforms.jax import (
        from_freq_to_wavelet as jax_from_freq_to_wavelet,
    )

    signal_wavelet_jax = jax_from_freq_to_wavelet(signal_freq, Nf=Nf, Nt=Nt)
    psd_wavelet_jax = evolutionary_psd_from_stationary_psd(
        psd=psd_freq.data,
        psd_f=psd_freq.freq,
        f_grid=signal_wavelet_jax.freq,
        t_grid=signal_wavelet_jax.time,
        dt=dt,
    )
    wdm_snr_jax = compute_snr(
        signal_wavelet_jax, signal_wavelet_jax, psd_wavelet_jax
    )
    assert np.isclose(snr, wdm_snr_jax, atol=0.5), f"{snr}!={wdm_snr_jax}"

    wdm_diff = signal_wavelet - signal_wavelet_jax

    ########################################
    # Part4: Plot
    ########################################

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    signal_wavelet.plot(ax=ax[0], absolute=True)
    signal_wavelet_jax.plot(ax=ax[1], absolute=True)
    wdm_diff.plot(ax=ax[2], absolute=True)
    ax[0].set_title(f"Numpy SNR={wdm_snr:.2f}")
    ax[1].set_title(f"Jax SNR={wdm_snr_jax:.2f}")
    ax[2].set_title("Difference")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/jax_vs_np.png")


def test_backend_loader():
    # temporarily set os.environ["PYWAVELET_JAX"] = "1"

    import pywavelet.backend

    os.environ["PYWAVELET_BACKEND"] = "jax"
    importlib.reload(pywavelet.backend)
    from pywavelet.backend import current_backend

    assert current_backend == "jax"
    os.environ["PYWAVELET_BACKEND"] = "numpy"

    importlib.reload(pywavelet.backend)
    from pywavelet.backend import current_backend

    assert current_backend == "numpy"
