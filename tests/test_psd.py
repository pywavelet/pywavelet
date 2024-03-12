import bilby
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.stats import norm
from pywavelet.utils.lvk import get_lvk_psd, get_lvk_psd_function
from pywavelet.utils.lisa import lisa_psd_func

from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms import from_wavelet_to_time, from_time_to_wavelet
from pywavelet.transforms.types import Wavelet, FrequencySeries, wavelet_dataset, TimeSeries
from pywavelet.psd import generate_noise_from_psd


def _make_plot(psd_wavelet, noise_wavelet, noise_pdgrm, psd_f, psd, fname):
    """Make a column of plots :
        - noise pdgrm (gray) + PSD (orange)
        - wavelet of noise
        - wavelet of PSD
        - ratio of noise / PSD
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))

    # plot the noise_pdgrm and the true psd
    _, ax = noise_pdgrm.plot(ax=axes[0], color="tab:blue", lw=0, marker=",", zorder=-10, alpha=0.5, label="Noise")
    ax.loglog(psd_f, psd, color="tab:orange", label="PSD", alpha=1, zorder=1)
    ax.legend()

    wavelet_plt_kwgs = dict(absolute=True, zscale="log", freq_scale="linear", freq_range=(min(psd_f), max(psd_f)))
    noise_wavelet.plot(ax=axes[1], **wavelet_plt_kwgs)
    psd_wavelet.plot(ax=axes[2], **wavelet_plt_kwgs)
    axes[1].set_title("Noise Wavelet")
    axes[2].set_title("PSD Wavelet")

    ratio = (noise_wavelet.data / ((psd_wavelet.data ** 2) * psd_wavelet.sample_rate / 2))
    bins = np.linspace(-5, 5, 100)
    axes[3].hist(ratio.flatten(), bins=bins)
    axes[3].set_title("Ratio of Noise / PSD")
    axes[3].plot(bins, norm.pdf(bins), ls="--")
    axes[3].set_xlabel(r"$\frac{\sqrt{|w_{nm}| \tau_s}}{S_{x}(f_m, t_n)}$")
    plt.tight_layout()
    fig.savefig(fname, dpi=300)
    return fig, axes


def test_wavelet_psd_from_stationary(plot_dir):
    """n: number of noise wavelets to take median of"""
    psd, psd_f = get_lvk_psd()
    psd_func = get_lvk_psd_function()

    # Wavelet data from noise
    noise_ts = generate_noise_from_psd(
        psd_func=psd_func,
        n_data=2 ** 17, fs=4028,
    )
    Nt = int(np.sqrt(len(noise_ts)))
    noise_pdgrm = FrequencySeries.from_time_series(noise_ts, min_freq=min(psd_f), max_freq=max(psd_f))
    noise_wavelet = from_time_to_wavelet(noise_ts, Nt=Nt, freq_range=(min(psd_f), max(psd_f)))

    # generate and plot the true PSD --> wavelet
    psd_wavelet: Wavelet = evolutionary_psd_from_stationary_psd(
        psd=np.sqrt(psd),
        psd_f=psd_f,
        f_grid=noise_wavelet.freq.data,
        t_grid=noise_wavelet.time.data,
    )

    # PLOTS
    mask = psd_f<1024
    _make_plot(
        psd_wavelet, noise_wavelet, noise_pdgrm, psd_f[mask], psd[mask], f"{plot_dir}/wavelet_psd_from_stationary.png"
    )
    plt.show()


def test_evolutionary_psd(plot_dir):
    # Analytical PSD S_LISA(f)*Amp(t) --> wavelet
    # Noise: S_LISA(f) --> timeseries * Amp(t) --> wavelet

    F_TRUE = 1e-5
    ALPHA_TRUE = 0.5

    def amp(t):
        return 1 + ALPHA_TRUE * np.cos(2 * np.pi * F_TRUE * t)

    tmax = 120 * 60 * 60  # Final time
    fs = 0.1  # Sampling rate
    delta_t = 1 / fs  # Sampling interval -- largely oversampling here.
    n_data = 2 ** int(np.log(tmax / delta_t) / np.log(2))

    q = int(np.log(n_data) / np.log(2))
    qf = int(q / 2) + 1
    Nt = 2 ** (q - qf)
    wavelet_kwgs = dict(Nt=Nt, nx=4.0, mult=32)
    df = Nt / (2 * tmax)

    noise = generate_noise_from_psd(lisa_psd_func, n_data=n_data, fs=fs)
    noise_mod = TimeSeries(data=noise.data * amp(noise.time.data), time=noise.time.data)

    noise_pdgrm = FrequencySeries.from_time_series(noise)
    noise_mod_pdgrm = FrequencySeries.from_time_series(noise_mod)
    true_psd = lisa_psd_func(noise_pdgrm.freq)

    noise_wavelet = from_time_to_wavelet(noise, **wavelet_kwgs)
    noise_mod_wavelet = from_time_to_wavelet(noise_mod, **wavelet_kwgs)
    freqs, times = noise_wavelet.freq, noise_wavelet.time
    # psd_wavelet
    psd_modulated_wavelet = wavelet_dataset(
        wavelet_data=np.sqrt(np.dot(
            (amp(times) ** 2).T, lisa_psd_func(freqs)
        ))
    )
    stf = np.sqrt(
        np.dot(np.asarray([modulation(tn) ** 2]).T, np.asarray([lisa_psd_func(fm)]))
    ).T
    analytical_wavelet = Wavelet.new(data=stf, time=tn, freq=fm)

    # plot freqseries
    fig = noise_pdgrm.plot(color="tab:gray", lw=0, marker=",", zorder=-10, alpha=0.5)
    ax = fig.gca()
    plt.loglog(noise_pdgrm.freq, true_psd, color="tab:orange", label="PSD", alpha=1, zorder=1)
    plt.show()

    noise_wavelet = from_time_to_wavelet(
        noise,
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
