import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def plot_wavelet_domain_grid(
    wavelet_data: np.ndarray,
    time_grid=None,
    freq_grid=None,
    Nt: int = None,
    Nf: int = None,
    ax=None,
    cmap="bwr",
) -> plt.Figure:
    """Plots the wavelet domain data (i.e. the wavelet amplitudes) as a 2D image."""
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    fig = ax.get_figure()

    # if Nt and Nf are not provided, infer them from the shape of the wavelet data
    Nt = wavelet_data.shape[0] if Nt is None else Nt
    Nf = wavelet_data.shape[1] if Nf is None else Nf

    if cmap == "bwr":
        norm = TwoSlopeNorm(
            vmin=np.min(wavelet_data), vcenter=0, vmax=np.max(wavelet_data)
        )
    else:
        norm = None

    extents = [0, Nt, 0, Nf]
    if time_grid is not None:
        extents[0] = time_grid[0]
        extents[1] = time_grid[-1]
    if freq_grid is not None:
        extents[2] = freq_grid[0]
        extents[3] = freq_grid[-1]

    im = ax.imshow(
        np.rot90(wavelet_data.T),
        aspect="auto",
        extent=extents,
        cmap=cmap,
        norm=norm,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Wavelet Amplitude")

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
    ax.set_xlabel(r"Time Bins [$\Delta T$=" + f"{1 / Nt:.2f}s, Nt={Nt}]")
    ax.set_ylabel(r"Freq Bins [$\Delta F$=" + f"{1 / Nf:.2f}Hz, Nf={Nf}]")
    plt.tight_layout()
    return fig
