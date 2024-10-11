from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.signal import spectrogram


def plot_wavelet_grid(
    wavelet_data: np.ndarray,
    time_grid=None,
    freq_grid=None,
    ax=None,
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=None,
    show_colorbar=True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D grid of wavelet coefficients.

    Parameters
    ----------
    wavelet_data : np.ndarray
        The wavelet freqseries to plot.

    time_grid : np.ndarray, optional
        The time grid for the wavelet freqseries.

    freq_grid : np.ndarray, optional
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

    z = np.rot90(wavelet_data.T)
    z = z if not absolute else np.abs(z)

    _norm = None
    _cmap = 'viridis'
    if not absolute:
        _cmap = "bwr"
        vmin = np.min(z)
        vmax = np.max(z)
        if vmin == vmax:
            vmin = -1
            vmax = 1
        if vmin == 0 :
            vmin = -vmax
        _norm = TwoSlopeNorm(
            vmin=vmin, vcenter=0, vmax=vmax
        )
    if zscale == "log":
        _norm = LogNorm(vmin=np.nanmin(z), vmax=np.nanmax(z))
    norm = kwargs.get("norm", _norm)
    cmap = kwargs.get("cmap", _cmap)

    extents = [0, Nt, 0, Nf]
    if time_grid is not None:
        extents[0] = time_grid[0]
        extents[1] = time_grid[-1]
    if freq_grid is not None:
        extents[2] = freq_grid[0]
        extents[3] = freq_grid[-1]

    im = ax.imshow(
        z, aspect="auto", extent=extents, cmap=cmap, norm=norm, interpolation="nearest"
    )
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        _cbar_label = "Absolute Wavelet Amplitude" if absolute else "Wavelet Amplitude"
        cl = kwargs.get("cbar_label", _cbar_label)
        cbar.set_label(cl)

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
    data: np.ndarray,
    freq: np.ndarray,
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
    data: np.ndarray, time: np.ndarray, ax=None, **kwargs
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
    timeseries_data: np.ndarray,
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
