from dataclasses import dataclass
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, TwoSlopeNorm
from xarray_dataclasses import AsDataArray, Coordof, Data, DataOptions, Name

from .common import FREQ, TIME, FreqAxis, TimeAxis, is_documented_by
from .plotting import plot_wavelet_grid

__all__ = ["Wavelet"]


class _Wavelet(xr.DataArray):
    """Custom DataArray."""

    __slots__ = ()

    @is_documented_by(plot_wavelet_grid)
    def plot(self, ax=None, *args, **kwargs) -> plt.Figure:
        """Plot the wavelet grid."""
        kwargs["time_grid"] = kwargs.get("time_grid", self.time.data)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq.data)
        return plot_wavelet_grid(
            wavelet_data=self.data, ax=ax, *args, **kwargs
        )

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
