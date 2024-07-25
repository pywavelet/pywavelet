import matplotlib.pyplot as plt
import numpy as np
import pytest

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.to_wavelets import from_time_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils.lisa import get_lisa_data
from pywavelet.utils.lvk import inject_signal_in_noise
from pywavelet.utils.snr import compute_snr


def test_lisa_snr(plot_dir):
    np.random.seed(1234)
    h_t, h_f, psd, snr = get_lisa_data()
    Nf = 256

    # FROM TIMESERIES
    data_from_timeseries = Data.from_timeseries(
        timeseries=h_t,
        Nf=Nf,
        mult=16,
    )

    data_from_freqseries = Data.from_frequencyseries(
        frequencyseries=h_f, Nf=Nf
    )

    psd_wavelet_time = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=data_from_timeseries.wavelet.freq,
        t_grid=data_from_timeseries.wavelet.time,
        dt=h_t.dt,
    )
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=data_from_freqseries.wavelet.freq,
        t_grid=data_from_freqseries.wavelet.time,
        dt=h_t.dt,
    )

    wavelet_snr_from_timeseries = np.sqrt(
        np.nansum(
            (data_from_timeseries.wavelet * data_from_timeseries.wavelet)
            / psd_wavelet_time
        )
    )
    wavelet_snr_from_freqseries = np.sqrt(
        np.nansum(
            (data_from_freqseries.wavelet * data_from_freqseries.wavelet)
            / psd_wavelet_freq
        )
    )

    print("Wavelet snr using timeseries = ", wavelet_snr_from_timeseries)
    print("Wavelet snr using freqseries = ", wavelet_snr_from_freqseries)
    print("snr is = ", snr)

    assert np.isclose(
        snr, wavelet_snr_from_timeseries, atol=1
    ), f"{snr} != {wavelet_snr_from_timeseries}, wavelet/freq snr = {snr / wavelet_snr_from_timeseries:.2f}"
    assert np.isclose(
        snr, wavelet_snr_from_freqseries, atol=1
    ), f"{snr} != {wavelet_snr_from_freqseries}, wavelet/freq snr = {snr / wavelet_snr_from_freqseries:.2f}"


def test_snr_lvk(plot_dir):
    Nf = 128
    h_f, psd, snr = inject_signal_in_noise(
        mc=30,
        noise=False,
    )
    h_f = FrequencySeries(data=h_f, freq=h_f.freq)
    data = Data.from_frequencyseries(
        h_f,
        Nf=Nf,
        mult=32,
    )
    fig, ax = data.plot_wavelet()
    fig.show()
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=data.wavelet.freq.data,
        t_grid=data.wavelet.time.data,
        dt=h_f.dt,
    )

    # cool line
    SNR2_wavelet = np.nansum((data.wavelet * data.wavelet) / psd_wavelet)
    print("wavelet_SNR is", SNR2_wavelet ** (1 / 2))
    assert np.isclose(snr, SNR2_wavelet**(1/2), atol=3)


# pytest parameterize decorator
@pytest.mark.parametrize(
    "f0, T, A, PSD_AMP, Nf",
    [
        (20, 1000, 1e-3, 1e-2, 16),
        (10, 1000, 1e-3, 1e-2, 32),
        (20, 1000, 1e-3, 1e-2, 16),
    ],
)
def test_toy_model_snr(f0, T, A, PSD_AMP, Nf):
    ########################################
    # Part1: Analytical SNR calculation
    ########################################
    dt = 0.5 / (
        2 * f0
    )  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array
    # round len(t) to the nearest power of 2
    t = t[: 2 ** int(np.log2(len(t)))]
    T = len(t) * dt

    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test
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
        SNR2_t, SNR2_t_analytical, atol=0.01
    ), f"{SNR2_t}!={SNR2_t_analytical}"
    assert np.isclose(
        SNR2_f, SNR2_t_analytical, atol=0.01
    ), f"{SNR2_f}!={SNR2_t_analytical}"

    ########################################
    # Part2: Wavelet domain
    ########################################
    ND = len(y)
    Nt = ND // Nf
    signal_timeseries = TimeSeries(y, t)
    signal_wavelet = from_time_to_wavelet(signal_timeseries, Nf=Nf, Nt=Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet.freq,
        t_grid=signal_wavelet.time,
        dt=dt,
    )
    wavelet_snr2 = compute_snr(signal_wavelet, psd_wavelet) ** 2

    assert np.isclose(
        wavelet_snr2, SNR2_t_analytical, atol=1e-2
    ), f"SNRs dont match {wavelet_snr2:.2f}!={SNR2_t_analytical:.2f} (factor:{SNR2_t_analytical/wavelet_snr2:.2f})"
