import bilby
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.stats import norm

from pywavelet.psd import (
    evolutionary_psd_from_stationary_psd,
    generate_noise_from_psd,
)
from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time
from pywavelet.transforms.types import (
    FrequencySeries,
    TimeSeries,
    Wavelet,
    wavelet_dataset,
)
from pywavelet.utils.lisa import lisa_psd_func
from pywavelet.utils.lvk import get_lvk_psd, get_lvk_psd_function


def _make_plot(psd_wavelet, noise_wavelet, noise_pdgrm, psd_f, psd, fname):
    """Make a column of plots :
    - noise pdgrm (gray) + PSD (orange)
    - wavelet of noise
    - wavelet of PSD
    - ratio of noise / PSD
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 14))

    # plot the noise_pdgrm and the true psd
    _, ax = noise_pdgrm.plot(
        ax=axes[0],
        color="tab:blue",
        lw=0,
        marker=".",
        zorder=-10,
        label="Noise",
    )
    ax.loglog(psd_f, psd, color="tab:orange", label="PSD", alpha=1, zorder=1)
    ax.set_xlim(min(psd_f), max(psd_f))
    ax.legend()

    wavelet_plt_kwgs = dict(
        absolute=True,
        zscale="log",
        freq_scale="log",
        freq_range=(min(psd_f), max(psd_f)),
    )
    noise_wavelet.plot(ax=axes[1], **wavelet_plt_kwgs)
    psd_wavelet.plot(ax=axes[2], **wavelet_plt_kwgs)
    axes[1].set_title("Noise Wavelet")
    axes[2].set_title("PSD Wavelet")

    d = noise_wavelet.data
    a = psd_wavelet.data**2
    fs = noise_pdgrm.sample_rate
    ratio = d / np.sqrt(a * fs / 2)
    # filter out ratios not in the frequency range
    ratio = ratio[
        (psd_wavelet.freq.data > min(psd_f))
        & (psd_wavelet.freq.data < max(psd_f)),
        :,
    ]
    # skip the start time and end time bins
    bins = np.linspace(-5, 5, 100)
    axes[3].hist(ratio.flatten(), bins=bins, density=True)
    axes[3].set_title("Ratio of Noise / PSD")
    axes[3].plot(bins, norm.pdf(bins), ls="--")
    axes[3].set_xlabel(r"$\frac{\sqrt{|w_{nm}| \tau_s}}{S_{x}(f_m, t_n)}$")
    plt.tight_layout()
    fig.savefig(fname, dpi=300)
    return fig, axes


def test_lvk_psd(plot_dir):
    """n: number of noise wavelets to take median of"""
    psd_func = get_lvk_psd_function()

    fs = 4096
    fmin, fmax = 20, fs / 2

    # Wavelet data from noise
    noise_ts = generate_noise_from_psd(
        psd_func=psd_func,
        n_data=2**17,
        fs=fs,
    )

    Nt = 128
    wavelet_kwgs = dict(Nt=Nt, nx=4.0, mult=32, freq_range=(fmin, fmax))
    noise_pdgrm = FrequencySeries.from_time_series(
        noise_ts, min_freq=fmin, max_freq=fmax
    )
    noise_wavelet = from_time_to_wavelet(noise_ts, **wavelet_kwgs)

    # generate and plot the true PSD --> wavelet
    lvk_psd = psd_func(noise_wavelet.freq)
    psd_grid = np.dot(np.ones((Nt, 1)), np.reshape(np.sqrt(lvk_psd), (1, -1)))
    psd_wavelet: Wavelet = wavelet_dataset(
        psd_grid,
        time_grid=noise_wavelet.time.data,
        freq_grid=noise_wavelet.freq.data,
    )

    # PLOTS
    # mask = (psd_f < fmax) & (psd_f > fmin)
    _make_plot(
        psd_wavelet,
        noise_wavelet,
        noise_pdgrm,
        noise_pdgrm.freq,
        psd_func(noise_pdgrm.freq),
        f"{plot_dir}/lvk_wavelet_psd.png",
    )


def test_lisa_psd(plot_dir):
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

    noise = generate_noise_from_psd(lisa_psd_func, n_data=n_data, fs=fs)
    noise_mod = TimeSeries(data=noise.data * amp(noise.time), time=noise.time)

    noise_pdgrm = FrequencySeries.from_time_series(noise)
    noise_mod_pdgrm = FrequencySeries.from_time_series(noise_mod)
    true_psd = lisa_psd_func(noise_pdgrm.freq)

    noise_wavelet = from_time_to_wavelet(noise, **wavelet_kwgs)
    noise_mod_wavelet = from_time_to_wavelet(noise_mod, **wavelet_kwgs)
    freqs, times = noise_wavelet.freq, noise_wavelet.time
    df = Nt / (2 * noise_pdgrm.duration)

    # repeat PSD Nt times along time axis
    lisa_psd = np.reshape(lisa_psd_func(freqs.data), (1, -1))
    amplitudes = np.reshape(amp(times.data) ** 2, (-1, 1))
    non_evol = np.dot(np.ones((Nt, 1)), lisa_psd)
    evol = np.dot(amplitudes, lisa_psd)
    psd_wavelet = wavelet_dataset(
        wavelet_data=np.sqrt(non_evol), time_grid=times, freq_grid=freqs
    )
    psd_modulated_wavelet = wavelet_dataset(
        wavelet_data=np.sqrt(evol), time_grid=times, freq_grid=freqs
    )

    mask = noise_mod_pdgrm.freq >= df
    _make_plot(
        psd_wavelet,
        noise_wavelet,
        noise_pdgrm,
        noise_pdgrm.freq[mask],
        true_psd[mask],
        f"{plot_dir}/lisa_wavelet_psd.png",
    )
    _make_plot(
        psd_modulated_wavelet,
        noise_mod_wavelet,
        noise_mod_pdgrm,
        noise_mod_pdgrm.freq[mask],
        true_psd[mask],
        f"{plot_dir}/lisa_wavelet_modulated_psd.png",
    )
