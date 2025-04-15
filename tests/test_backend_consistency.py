from scipy.signal import chirp
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from typing import Tuple, Union

jax.config.update('jax_enable_x64', True)

from matplotlib.pyplot import GridSpec
from pywavelet.types import TimeSeries, FrequencySeries, Wavelet
from pywavelet.transforms.numpy import from_freq_to_wavelet, from_wavelet_to_freq
from pywavelet.transforms.jax import from_freq_to_wavelet as jax_from_freq_to_wavelet
from pywavelet.transforms.jax import from_wavelet_to_freq as jax_from_wavelet_to_freq
from dataclasses import dataclass


@dataclass
class RoundtripData:
    freq: FrequencySeries
    wavelet: Wavelet
    reconstructed: FrequencySeries


def generate_monochrome_signal(
        t: np.ndarray, f0: float = 50
) -> FrequencySeries:
    dt = t[1] - t[0]
    fs = 1 / dt
    nyquist = fs / 2
    assert (f0 < nyquist), f"f0 [{f0:.2f} Hz] must be less than f_nyquist [{nyquist:2f} Hz]."
    y = np.sin(2 * np.pi * f0 * t)
    return TimeSeries(data=y, time=t).to_frequencyseries()


def _plot_residuals(ax, residuals):
    ax.hist(residuals, bins=100)
    # add textbox of mean and std
    mean = residuals.mean()
    std = residuals.std()
    textstr = f"$\mu={mean:.1E}$\n$\sigma={std:.1E}$"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Count")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    return ax


def _plot_row(label: str, data: RoundtripData, axes):
    f = data.freq.freq
    hf = data.freq.data
    hf_new = data.reconstructed.data
    _ = data.freq.plot_periodogram(ax=axes[0], color="black", label="Original")
    _ = data.reconstructed.plot_periodogram(ax=axes[0], ls="--", color="red", label="Reconstructed")
    axes[0].legend()
    axes[0].scatter(f, np.abs(hf_new) ** 2, color='red')
    axes[0].scatter(f, np.abs(hf) ** 2, color='black')
    _ = data.wavelet.plot(ax=axes[1], absolute=True, cmap="Reds")
    _ = _plot_residuals(axes[2], np.abs(hf - hf_new))
    axes[0].set_title(f"{label} Reconstructed")
    axes[1].set_title(f"{label} WDM")
    axes[2].set_title(f"{label} Recon Residuals")


def plot(
        np_data: RoundtripData,
        jax_data: RoundtripData,
        plot_fn: str,
) -> None:
    # make a gridspec 3 rows 3 columns (last row is full width)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, :]),
    ]

    _plot_row('NUMPY', np_data, [axes[0], axes[1], axes[2]])
    _plot_row('JAX', jax_data, [axes[3], axes[4], axes[5]])

    w_diff = np_data.wavelet - jax_data.wavelet
    err = np.sum(np.abs(w_diff.data))
    w_diff.plot(ax=axes[-1], label=f'err={err:.2f}')
    fig.tight_layout()
    fig.savefig(plot_fn)
    plt.close(fig)


def test_jax_and_numpy(plot_dir):
    # Sizes
    dt = 1 / 32
    Nt, Nf = 2 ** 3, 2 ** 3
    ND = Nt * Nf

    # time grid
    ts = np.arange(0, ND) * dt
    h_freq = generate_monochrome_signal(ts, f0=10.0)

    # using numpy
    h_wavelet = from_freq_to_wavelet(h_freq, Nf=Nf, Nt=Nt)
    h_reconstructed = from_wavelet_to_freq(h_wavelet, dt=dt)
    np_data = RoundtripData(freq=h_freq, wavelet=h_wavelet, reconstructed=h_reconstructed)

    # using JAX
    jax_h_wavelet = jax_from_freq_to_wavelet(h_freq, Nf=Nf, Nt=Nt)
    jax_h_reconstructed = jax_from_wavelet_to_freq(jax_h_wavelet, dt=dt)
    jax_data = RoundtripData(freq=h_freq, wavelet=jax_h_wavelet, reconstructed=jax_h_reconstructed)

    # Plotting
    plot(
        np_data,
        jax_data,
        plot_fn=f"{plot_dir}/jax_vs_numpy_roundtrip.png",
    )
