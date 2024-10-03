from typing import Tuple

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, TwoSlopeNorm

import numpy as np
import jax.numpy as jnp

from scipy.signal.spectral import spectrogram



def plot_wavelet_grid(
    wavelet_data: jnp.ndarray,
    time_grid=None,
    freq_grid=None,
    ax=None,
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=None,
    norm=None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D grid of wavelet coefficients.

    Parameters
    ----------
    wavelet_data : jnp.ndarray
        The wavelet freqseries to plot.

    time_grid : jnp.ndarray, optional
        The time grid for the wavelet freqseries.

    freq_grid : jnp.ndarray, optional
        The frequency grid for the wavelet freqseries.

    ax : plt.Axes, optional
        The axes to plot on.

    zscale : str, optional
        The scale for the colorbar.

    freq_scale : str, optional
        The scale for the frequency axis.

    absolute : bool, optional
        Whether to plot the absolute value of the wavelet freqseries.

    freq_range : Tuple[float, float], optional
        The frequency range to plot.

    norm : matplotlib.colors.Normalize, optional
        The normalization for the colorbar. If None, a default normalization is used.
        Useful for comparing different plots.

    kwargs : dict, optional
        Additional keyword arguments for the plot.

    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    fig = ax.get_figure()

    Nf, Nt = wavelet_data.shape
    assert Nf == len(freq_grid), f"Nf={Nf} != len(freq_grid)={len(freq_grid)}"
    assert Nt == len(time_grid), f"Nt={Nt} != len(time_grid)={len(time_grid)}"

    z = jnp.rot90(wavelet_data.T)
    if absolute:
        z = jnp.abs(z)

    if not absolute:
        try:
            cmap = "bwr"
            norm = TwoSlopeNorm(
                vmin=jnp.min(wavelet_data), vcenter=0, vmax=jnp.max(wavelet_data)
            )
        except Exception:
            cmap = "viridis"
    else:
        cmap = kwargs.get("cmap", "viridis")

    if zscale == "log":
        norm = LogNorm(vmin=jnp.nanmin(z), vmax=jnp.nanmax(z))

    extents = [0, Nt, 0, Nf]
    if time_grid is not None:
        extents[0] = time_grid[0]
        extents[1] = time_grid[-1]
    if freq_grid is not None:
        extents[2] = freq_grid[0]
        extents[3] = freq_grid[-1]

    im = ax.imshow(z, aspect="auto", extent=extents, cmap=cmap, norm=norm)
    try:
        cbar = plt.colorbar(im, ax=ax)
        cl = "Absolute Wavelet Amplitude" if absolute else "Wavelet Amplitude"
        cbar.set_label(cl)
    except Exception:
        pass

    # add a text box with the Nt and Nf values
    ax.text(
        0.05,
        0.95,
        f"{Nt}x{Nf}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=None, alpha=0.2),
    )
    ax.set_yscale(freq_scale)
    ax.set_xlabel(
        r"Time Bins [$\Delta T$=" + f"{1 / Nt:.4f}s, Nt={Nt}]", fontsize=15
    )
    ax.set_ylabel(
        r"Freq Bins [$\Delta F$=" + f"{1 / Nf:.4f}Hz, Nf={Nf}]", fontsize=15
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if freq_range is not None:
        ax.set_ylim(freq_range)

    plt.tight_layout()
    return fig, ax


def plot_freqseries(
    data: jnp.ndarray,
    freq: jnp.ndarray,
    nyquist_frequency: float,
    ax=None,
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(freq, data, **kwargs)
    ax.set_xlabel("Frequency Bin [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-nyquist_frequency, nyquist_frequency)
    return ax.figure, ax


def plot_periodogram(
    data: np.ndarray,
    freq: np.ndarray,
    nyquist_frequency: float,
    ax=None,
    **kwargs,
):
    if ax == None:
        fig, ax = plt.subplots()

    ax.loglog(freq, np.abs(data) ** 2, **kwargs)
    flow = np.min(np.abs(freq))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Periodigram")
    ax.set_xlim(left=flow, right=nyquist_frequency)
    return ax.figure, ax


def plot_timeseries(
    data: jnp.ndarray, time: jnp.ndarray, ax=None, **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Custom method."""
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(time, data, **kwargs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(left=time[0], right=time[-1])
    return ax.figure, ax


def plot_spectrogram(
    timeseries_data: jnp.ndarray,
    fs: float,
    ax=None,
    spec_kwargs={},
    plot_kwargs={},
) -> Tuple[plt.Figure, plt.Axes]:
    f, t, Sxx = spectrogram(timeseries_data, fs=fs, **spec_kwargs)
    if ax == None:
        fig, ax = plt.subplots()

    if "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = "Reds"

    cm = ax.pcolormesh(t, f, Sxx, shading="nearest", **plot_kwargs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(top=fs / 2.0)
    cbar = plt.colorbar(cm, ax=ax)
    cbar.set_label("Spectrogram Amplitude")
    return ax.figure, ax
