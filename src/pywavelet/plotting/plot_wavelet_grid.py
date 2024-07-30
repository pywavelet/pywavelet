from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm


def plot_wavelet_grid(
    wavelet_data: np.ndarray,
    time_grid=None,
    freq_grid=None,
    ax=None,
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 2D grid of wavelet coefficients.

    Parameters
    ----------
    wavelet_data : np.ndarray
        The wavelet data to plot.

    time_grid : np.ndarray, optional
        The time grid for the wavelet data.

    freq_grid : np.ndarray, optional
        The frequency grid for the wavelet data.

    ax : plt.Axes, optional
        The axes to plot on.

    zscale : str, optional
        The scale for the colorbar.

    freq_scale : str, optional
        The scale for the frequency axis.

    absolute : bool, optional
        Whether to plot the absolute value of the wavelet data.

    freq_range : Tuple[float, float], optional
        The frequency range to plot.

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
    if absolute:
        z = np.abs(z)

    norm = None
    if not absolute:
        try:
            cmap = "bwr"
            norm = TwoSlopeNorm(
                vmin=np.min(wavelet_data), vcenter=0, vmax=np.max(wavelet_data)
            )
        except Exception:
            cmap = "viridis"
    else:
        cmap = kwargs.get("cmap", "viridis")

    if zscale == "log":
        norm = LogNorm(vmin=np.nanmin(z), vmax=np.nanmax(z))

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
