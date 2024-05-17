import matplotlib.pyplot as plt
import numpy as np
from gw_utils import DT, DURATION, inject_signal_in_noise

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries
from pywavelet.utils.lisa import get_lisa_data
from pywavelet.utils.snr import compute_frequency_optimal_snr, compute_snr

Nf, Nt = 512, 256
ND = Nf * Nt
T_BINWIDTH = DURATION / Nt
F_BINWIDTH = 1 / 2 * T_BINWIDTH
FMAX = 1 / (2 * DT)

T_GRID = np.linspace(0, DURATION, Nt)
F_GRID = np.linspace(0, FMAX, Nf)


def test_snr_lvk(plot_dir):
    distance = 100
    h_time, timeseries_snr, ifo = inject_signal_in_noise(
        mc=30, q=1, distance=distance, noise=False
    )
    data = Data.from_timeseries(
        h_time,
        minimum_frequency=ifo.minimum_frequency,
        maximum_frequency=ifo.maximum_frequency,
        Nf=Nf,
        mult=32,
    )
    mask = ifo.strain_data.frequency_mask

    psd, psd_f = (
        ifo.power_spectral_density_array[mask],
        ifo.frequency_array[mask],
    )

    custom_timeseries_snr = compute_frequency_optimal_snr(
        h_freq=ifo.frequency_domain_strain[mask],
        psd=ifo.power_spectral_density_array[mask],
        duration=ifo.duration,
    )
    assert timeseries_snr == custom_timeseries_snr

    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=ifo.power_spectral_density_array[mask],
        psd_f=ifo.frequency_array[mask],
        f_grid=data.wavelet.freq.data,
        t_grid=data.wavelet.time.data,
    )

    wavelet_snr = compute_snr(data.wavelet, psd_wavelet)
    # assert within +/- 10 of each other
    assert np.isclose(timeseries_snr, wavelet_snr, atol=10)

    __make_plots(
        data,
        psd_wavelet,
        psd,
        psd_f,
        f"{plot_dir}/snr_lvk.png",
        labels=[f"snr={timeseries_snr:.2f}", f"wavelet-snr={wavelet_snr:.2f}"],
    )


def test_lisa_snr(plot_dir):
    np.random.seed(1234)

    h_signal_t, t, h_signal_f, f, psd_f, snr = get_lisa_data()
    dt = t[1] - t[0]
    Nf = 256

    h_time = TimeSeries(data=h_signal_t, time=t)
    data = Data.from_timeseries(
        timeseries=h_time,
        Nf=Nf,
        mult=16,
        minimum_frequency=9**-4,
        maximum_frequency=0.02,
    )

    h_wavelet = from_time_to_wavelet(h_time, Nt=Nt) * np.sqrt(2) * dt

    # h_wavelet = from_time_to_wavelet(h_time, Nt=Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd_f,
        psd_f=f,
        f_grid=h_wavelet.freq.data,
        t_grid=h_wavelet.time.data,
    )

    compute_frequency_optimal_snr(
        h_freq=h_signal_f, psd=psd_f, duration=h_time.duration
    )
    wavelet_snr = compute_snr(h_wavelet, psd_wavelet)
    __make_plots(
        data,
        psd_wavelet,
        psd_f,
        f,
        f"{plot_dir}/snr_lisa.png",
        log=False,
        labels=[f"snr={snr:.0f}", f"wavelet-snr={wavelet_snr:.0f}"],
    )
    assert np.isclose(
        snr, wavelet_snr, atol=10
    ), f"{snr} != {wavelet_snr}, wavelet/freq snr = {snr / wavelet_snr:.2f}"


def __make_plots(data, psd_wavelet, psd, psd_f, fname, log=True, labels=[]):
    fig, axes = plt.subplots(5, 1, figsize=(5, 15))
    data.plot_all(
        axes=axes,
    )
    axes[1].plot(psd_f, psd, label="PSD")

    axes[1].set_ylim(bottom=min(psd) * 0.1)
    psd_wavelet.plot(
        ax=axes[-1], absolute=True, zscale="log", freq_scale="log"
    )
    axes[-1].set_ylim(data.minimum_frequency, data.maximum_frequency)
    if log:
        axes[2].set_yscale("log")
        axes[3].set_yscale("log")

    lblkwg = dict(
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.1),
    )
    if labels:
        # top right corner label with frame
        axes[1].text(
            0.05, 0.1, labels[0], transform=axes[1].transAxes, **lblkwg
        )
        axes[3].text(
            0.05, 0.1, labels[1], transform=axes[3].transAxes, **lblkwg
        )
    plt.tight_layout()
    fig.savefig(fname, dpi=300)


def test_sine_snr():
    true_f = 10
    fs = 20 * 3
    dt = 1 / fs
    ND = 4098
    time = np.linspace(0, ND * dt, ND)
    duration = ND * dt
    signal_t = np.sin(2 * np.pi * true_f * time)
    signal_f = np.fft.fft(signal_t)
    freq = np.fft.fftfreq(ND, dt)

    psd_f = np.ones(len(signal_f))
    snr = compute_frequency_optimal_snr(
        signal_f[: ND // 2], psd_f[: ND // 2], duration=duration
    )
    signal_t = TimeSeries(
        data=signal_t,
        time=time,
    )

    # analytical SNR for flat white noise in time domain:
    # sin**2 ~ 0.5
    sigma_sqr = 0.5 * dt
    analytical_snr = np.sqrt(np.sum(signal_t.data**2 / sigma_sqr))

    assert np.isclose(
        analytical_snr - snr, 0, atol=1e-2
    ), f"{analytical_snr} != {snr}"

    signal_wavelet = from_time_to_wavelet(signal_t, Nf=64)

    signal_t.plot()
    plt.show()

    signal_wavelet.plot()
    plt.show()

    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd_f,
        psd_f=np.arange(len(psd_f)),
        f_grid=signal_wavelet.freq.data,
        t_grid=signal_wavelet.time.data,
    )
    # psd_wavelet.plot()
    # plt.show()

    wavelet_snr = compute_snr(signal_wavelet, psd_wavelet)

    assert np.isclose(
        snr, wavelet_snr, atol=10
    ), f"{snr} != {wavelet_snr}, wavelet/freq snr = {snr / wavelet_snr:.2f}"
