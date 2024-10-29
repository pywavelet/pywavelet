import warnings
from typing import Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.signal import spectrogram

MIN_S = 60
HOUR_S = 60 * MIN_S
DAY_S = 24 * HOUR_S

def plot_wavelet_grid(
    wavelet_data: np.ndarray,
    time_grid: np.ndarray,
    freq_grid: np.ndarray,
    ax: Optional[plt.Axes] = None,
    zscale: str = "linear",
    freq_scale: str = "linear",
    absolute: bool = False,
    freq_range: Optional[Tuple[float, float]] = None,
    show_colorbar: bool = True,
    cmap: Optional[str] = None,
    norm: Optional[Union[LogNorm, TwoSlopeNorm]] = None,
    cbar_label: Optional[str] = None,
    nan_color: Optional[str] = "black",
    detailed_axes:bool = False,
    show_gridinfo:bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D grid of wavelet coefficients.

    Parameters
    ----------
    wavelet_data : np.ndarray
        A 2D array containing the wavelet coefficients with shape (Nf, Nt),
        where Nf is the number of frequency bins and Nt is the number of time bins.

    time_grid : np.ndarray, optional
        1D array of time values corresponding to the time bins. If None, uses np.arange(Nt).

    freq_grid : np.ndarray, optional
        1D array of frequency values corresponding to the frequency bins. If None, uses np.arange(Nf).

    ax : plt.Axes, optional
        Matplotlib Axes object to plot on. If None, creates a new figure and axes.

    zscale : str, optional
        Scale for the color mapping. Options are 'linear' or 'log'. Default is 'linear'.

    freq_scale : str, optional
        Scale for the frequency axis. Options are 'linear' or 'log'. Default is 'linear'.

    absolute : bool, optional
        If True, plots the absolute value of the wavelet coefficients. Default is False.

    freq_range : tuple of float, optional
        Tuple specifying the (min, max) frequency range to display. If None, displays the full range.

    show_colorbar : bool, optional
        If True, displays a colorbar next to the plot. Default is True.

    cmap : str, optional
        Colormap to use for the plot. If None, uses 'viridis' for absolute values or 'bwr' for signed values.

    norm : matplotlib.colors.Normalize, optional
        Normalization instance to scale data values. If None, a suitable normalization is chosen based on `zscale`.

    cbar_label : str, optional
        Label for the colorbar. If None, a default label is used based on the `absolute` parameter.

    nan_color : str, optional
        Color to use for NaN values. Default is 'black'.

    **kwargs
        Additional keyword arguments passed to `ax.imshow()`.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects of the plot.

    Raises
    ------
    ValueError
        If the dimensions of `wavelet_data` do not match the lengths of `freq_grid` and `time_grid`.
    """

    # Determine the dimensions of the data
    Nf, Nt = wavelet_data.shape

    # Validate the dimensions
    if (Nf, Nt) != (len(freq_grid), len(time_grid)):
        raise ValueError(f"Wavelet shape {Nf, Nt} does not match provided grids {(len(freq_grid), len(time_grid))}.")

    # Prepare the data for plotting
    z = wavelet_data.copy()
    if absolute:
        z = np.abs(z)

    # Determine normalization and colormap
    if norm is None:
        try:
            if np.all(np.isnan(z)):
                raise ValueError("All wavelet data is NaN.")
            if zscale == "log":
                norm = LogNorm(vmin=np.nanmin(z[z > 0]), vmax=np.nanmax(z))
            elif not absolute:
                vmin, vmax = np.nanmin(z), np.nanmax(z)
                vcenter = 0.0
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            else:
                norm = None  # Default linear scaling
        except Exception as e:
            warnings.warn(f"Error in determining normalization: {e}. Using default linear scaling.")
            norm = None

    if cmap is None:
        cmap = "viridis" if absolute else "bwr"
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color=nan_color)

    # Set up the plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Plot the data
    im = ax.imshow(
        z,
        aspect="auto",
        extent=[time_grid[0],time_grid[-1], freq_grid[0], freq_grid[-1]],
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        **kwargs,
    )


    # Configure axes scales
    ax.set_yscale(freq_scale)
    _fmt_time_axis(time_grid, ax)


    ax.set_ylabel("Frequency [Hz]")
    if detailed_axes:
        ax.set_xlabel(r"Time Bins [$\Delta T$=" + f"{1 / Nt:.4f}s, Nt={Nt}]")
        ax.set_ylabel(r"Freq Bins [$\Delta F$=" + f"{1 / Nf:.4f}Hz, Nf={Nf}]")

    label = kwargs.get("label", "")
    label += f"{Nt}x{Nf}" if show_gridinfo else ""
    if label:
        ax.text(
            0.05,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=None, alpha=0.2),
        )


    # Set frequency range if specified
    if freq_range is not None:
        ax.set_ylim(freq_range)

    # Add colorbar if requested
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label is None:
            cbar_label = "Absolute Wavelet Amplitude" if absolute else "Wavelet Amplitude"
        cbar.set_label(cbar_label)

    # Adjust layout
    fig.tight_layout()

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
    ax.set_xlim(0, nyquist_frequency)
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

    ax.set_ylabel("Amplitude")
    ax.set_xlim(left=time[0], right=time[-1])

    _fmt_time_axis(time, ax)

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

    _fmt_time_axis(t, ax)


    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(top=fs / 2.0)
    cbar = plt.colorbar(cm, ax=ax)
    cbar.set_label("Spectrogram Amplitude")
    return ax.figure, ax



def _fmt_time_axis(t, axes):
    if t[-1] > DAY_S:  # If time goes beyond a day
        axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / DAY_S:.1f}"))
        axes.set_xlabel("Time [days]")
    elif t[-1] > HOUR_S:  # If time goes beyond an hour
        axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / HOUR_S:.1f}"))
        axes.set_xlabel("Time [hr]")
    elif t[-1] > MIN_S:  # If time goes beyond a minute
        axes.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / MIN_S:.1f}"))
        axes.set_xlabel("Time [min]")
    else:
        axes.set_xlabel("Time [s]")
