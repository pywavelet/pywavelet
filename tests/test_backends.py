import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils import cuda_available

from pywavelet import set_backend
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd

import logging

logger = logging.getLogger("pywavelet")


@pytest.mark.parametrize("backend", ["jax", "cupy"])
def test_toy_model_snr(backend, plot_dir):
    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")

    f0 = 20
    dt = 0.0125
    A = 2
    Nt = 128
    Nf = 256
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    PSD_AMP = 1

    true_wdm_amp = A * np.sqrt(2 * Nf)

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
    np.testing.assert_allclose(true_wdm_amp, signal_wavelet.data.max(), atol=1e-2)

    ########################################
    # Part3: Wavelet domain (other backend)
    ########################################
    signal_wavelet_new = None
    if backend == "jax":
        logger.info("TESTING JAX")
        import jax.numpy as xp

        from pywavelet.transforms.jax import (
            from_freq_to_wavelet as jax_from_freq_to_wavelet,
        )

        signal_freq.data = xp.array(signal_freq.data)
        signal_wavelet_new = jax_from_freq_to_wavelet(
            signal_freq, Nf=Nf, Nt=Nt
        )
        # convert back to numpy
        signal_wavelet_new.data = np.array(signal_wavelet_new.data)
        assert isinstance(signal_wavelet_new.data, np.ndarray)
        logger.info("JAX TEST COMPLETE")



    elif backend == "cupy":
        logger.info("TESTING CUPY")
        import cupy as xp

        from pywavelet.transforms.cupy import (
            from_freq_to_wavelet as cupy_from_freq_to_wavelet,
        )

        signal_freq.data = xp.array(signal_freq.data)
        signal_wavelet_new = cupy_from_freq_to_wavelet(
            signal_freq, Nf=Nf, Nt=Nt
        )
        # convert back to numpy
        signal_wavelet_new.data = np.array(signal_wavelet_new.data.get())
        assert isinstance(signal_wavelet_new.data, np.ndarray)
        logger.info("CUPY TEST COMPLETE")

    np.testing.assert_allclose(true_wdm_amp, signal_wavelet_new.data.max(), atol=1e-2)
    wdm_snr_jax = compute_snr(
        signal_wavelet_new, signal_wavelet_new, psd_wavelet
    )
    assert np.isclose(snr, wdm_snr_jax, atol=0.5), f"{snr}!={wdm_snr_jax}"

    wdm_diff = signal_wavelet - signal_wavelet_new

    ########################################
    # Part4: Plot
    ########################################

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    signal_wavelet.plot(ax=ax[0], absolute=True, cmap='Grays')
    signal_wavelet_new.plot(ax=ax[1], absolute=True, cmap='Grays')
    wdm_diff.plot(ax=ax[2], absolute=True, cmap='Grays')
    ax[0].set_title(f"Numpy SNR={wdm_snr:.2f}")
    ax[1].set_title(f"{backend} SNR={wdm_snr_jax:.2f}")
    ax[2].set_title("Difference")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{backend}_vs_np.png")


def test_backend_loader():
    backends = ["jax", "cupy"]

    for backend in backends:
        set_backend(backend)
        from pywavelet.backend import current_backend, xp

        assert current_backend == backend

    set_backend("numpy")
