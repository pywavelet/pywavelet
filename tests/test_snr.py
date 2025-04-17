import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import monochromatic_wnm
from utils import cuda_available

from pywavelet import set_backend
from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
from pywavelet.types import FrequencySeries, TimeSeries
from pywavelet.types.wavelet_bins import compute_bins
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd


def test_toy_model_snr(plot_dir):
    f0 = 20
    dt = 0.0125
    A = 2
    Nt = 128
    Nf = 256
    T = Nt * Nf * dt
    ND = Nt * Nf
    t = np.arange(0, ND) * dt
    PSD_AMP = 1
    ########################################
    # Part1: Analytical SNR calculation
    #######################################

    # Eq 21
    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test

    # makes the freq -> [-0.5,...0,... 0.5] Hz
    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency
    y_fft = dt * np.fft.fftshift(
        np.fft.fft(y)
    )  # continuous time fourier transform [seconds]

    PSD = PSD_AMP * np.ones(len(freq))  # PSD of the noise

    # Compute the SNRs
    SNR2_f = 2 * np.sum(abs(y_fft) ** 2 / PSD) * df
    SNR2_t = 2 * dt * np.sum(abs(y) ** 2 / PSD)
    SNR2_t_analytical = (A**2) * T / PSD[0]

    assert np.isclose(
        SNR2_t, SNR2_t_analytical, atol=0.5
    ), f"{SNR2_t}!={SNR2_t_analytical}"
    assert np.isclose(
        SNR2_f, SNR2_t_analytical, atol=0.5
    ), f"{SNR2_f}!={SNR2_t_analytical}"

    ########################################
    # Part2: Wavelet domain
    ########################################
    ND = len(y)
    Nt = ND // Nf
    assert Nt > 1, f"Nt={Nt} must be greater than 1 (ND={ND}, Nf={Nf})"
    signal_timeseries = TimeSeries(y, t)

    # time --> wavelet
    signal_wavelet = from_time_to_wavelet(signal_timeseries, Nf=Nf, Nt=Nt)
    psd_wavelet_time = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet.freq,
        t_grid=signal_wavelet.time,
        dt=dt,
    )
    time2wavelet_snr2 = (
        compute_snr(signal_wavelet, signal_wavelet, psd_wavelet_time) ** 2
    )

    # freq --> wavelet
    signal_freq = signal_timeseries.to_frequencyseries()
    psd_freq = FrequencySeries(
        PSD_AMP * np.ones(len(signal_freq)), signal_freq.freq
    )
    np.testing.assert_almost_equal(
        signal_freq.optimal_snr(psd_freq) ** 2, SNR2_f
    )

    # assert len(signal_freq) == (ND // 2 ) + 1 , f"Not one sided spectrum {len(signal_freq)}!={(ND // 2 ) + 1}"
    signal_wavelet_f = from_freq_to_wavelet(signal_freq, Nf=Nf, Nt=Nt)
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_f.freq,
        t_grid=signal_wavelet_f.time,
        dt=dt,
    )
    freq2wavelet_snr2 = (
        compute_snr(signal_wavelet_f, signal_wavelet_f, psd_wavelet_freq) ** 2
    )

    # analytical wavelet
    analytical_wnm = monochromatic_wnm(f0, dt, A, Nt, Nf)
    analytical_wavelet_snr2 = (
        compute_snr(analytical_wnm, analytical_wnm, psd_wavelet_time) ** 2
    )

    assert np.isclose(
        time2wavelet_snr2, SNR2_t, atol=1e-2
    ), f"SNRs dont match {time2wavelet_snr2:.2f}!={SNR2_t:.2f} (factor:{SNR2_t / time2wavelet_snr2:.2f})"
    assert np.isclose(
        freq2wavelet_snr2, SNR2_t, atol=1e-2
    ), f"SNRs dont match {freq2wavelet_snr2:.2f}!={SNR2_t:.2f} (factor:{SNR2_t / freq2wavelet_snr2:.2f})"
    assert np.isclose(
        analytical_wavelet_snr2, SNR2_t, atol=1e-2
    ), f"SNRs dont match {analytical_wavelet_snr2:.2f}!={SNR2_t:.2f} (factor:{SNR2_t / analytical_wavelet_snr2:.2f})"

    #### PLOTTING
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    signal_wavelet.plot(ax=axes[0], absolute=False)
    signal_wavelet_f.plot(ax=axes[1], absolute=False)
    analytical_wnm.plot(ax=axes[2], absolute=False)
    axes[0].set_title("t->wdm")
    axes[1].set_title("f->wdm")
    axes[2].set_title("analytical wdm")
    for ax in axes:
        ax.set_xlim(0, 50)
        # ax.set_ylim(f0+2.5, f0-2.5)
    #     ax.set_xlabel("Time [s]")
    #     ax.set_ylabel("")
    # axes[0].set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/snr_comparison.png")


import logging

logger = logging.getLogger("pywavelet")


@pytest.mark.parametrize("backend", ["jax", "cupy"])
def test_toy_model_snr2(backend, plot_dir):
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
    np.testing.assert_allclose(
        true_wdm_amp, signal_wavelet.data.max(), atol=1e-2
    )

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

    np.testing.assert_allclose(
        true_wdm_amp, signal_wavelet_new.data.max(), atol=1e-2
    )
    wdm_snr_jax = compute_snr(
        signal_wavelet_new, signal_wavelet_new, psd_wavelet
    )
    assert np.isclose(snr, wdm_snr_jax, atol=0.5), f"{snr}!={wdm_snr_jax}"

    wdm_diff = signal_wavelet - signal_wavelet_new

    ########################################
    # Part4: Plot
    ########################################

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    signal_wavelet.plot(ax=ax[0], absolute=True, cmap="Grays")
    signal_wavelet_new.plot(ax=ax[1], absolute=True, cmap="Grays")
    wdm_diff.plot(ax=ax[2], absolute=True, cmap="Grays")
    ax[0].set_title(f"Numpy SNR={wdm_snr:.2f}")
    ax[1].set_title(f"{backend} SNR={wdm_snr_jax:.2f}")
    ax[2].set_title("Difference")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{backend}_vs_np.png")
