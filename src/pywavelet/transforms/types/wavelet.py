from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.signal import spectrogram
from xarray_dataclasses import (
    AsDataArray,
    Attr,
    Coordof,
    Data,
    DataOptions,
    Name,
)

from .common import FREQ, TIME, FreqAxis, TimeAxis

__all__ = ["Wavelet"]


class _Wavelet(xr.DataArray):
    """Custom DataArray."""

    __slots__ = ()

    def plot(self, ax=None, *args, **kwargs) -> plt.Figure:
        """Custom method."""
        kwargs["time_grid"] = kwargs.get("time_grid", self.time.data)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq.data)
        return plot_wavelet_grid(self.data, ax=ax, *args, **kwargs)

    @property
    def Nt(self) -> int:
        """Number of time bins."""
        return len(self.time)

    @property
    def Nf(self) -> int:
        """Number of frequency bins."""
        return len(self.freq)

    @property
    def ND(self) -> int:
        return self.Nt * self.Nf

    @property
    def delta_T(self):
        # TODO: call these TIME BINs not time --> reserve time for the 'time-domain' time axis
        return self.time[1] - self.time[0]

    @property
    def delta_F(self):
        return 1 / (2 * self.delta_T)

    @property
    def duration(self) -> float:
        return self.Nt * self.delta_T

    @property
    def delta_t(self) -> float:
        return self.duration / self.ND

    @property
    def delta_f(self) -> float:
        return 1 / (2 * self.delta_t)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the wavelet grid."""
        return self.data.shape

    @property
    def sample_rate(self) -> float:
        return 1 / self.delta_t

    @property
    def fs(self):
        return self.sample_rate

    @property
    def nyquist_frequency(self) -> float:
        return self.sample_rate / 2


@dataclass
class Wavelet(AsDataArray):
    data: Data[Tuple[FREQ, TIME], float]
    time: Coordof[TimeAxis] = 0.0
    freq: Coordof[FreqAxis] = 0.0
    name: Name[str] = "Wavelet Amplitude"

    __dataoptions__ = DataOptions(_Wavelet)

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        time_grid: Optional[Union[np.array, None]] = None,
        freq_grid: Optional[Union[np.array, None]] = None,
        freq_range: Optional[Union[np.array, None]] = None,
        time_range: Optional[Union[np.array, None]] = None,
    ) -> "Wavelet":
        """Create a dataset with wavelet coefficients.

        Parameters
        ----------
        data : np.ndarray
            Wavelet coefficients.

        Returns
        -------
        Wavelet

        """
        w = cls.new(data.T, time=time_grid, freq=freq_grid)

        if freq_range is not None:
            w = w.sel(freq=slice(*freq_range))
        if time_range is not None:
            w = w.sel(time=slice(*time_range))
        return w


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
