from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray_dataclasses import (
    AsDataArray,
    Attr,
    Coord,
    Coordof,
    Data,
    DataOptions,
    Name,
)

from pywavelet.plotting import plot_wavelet_domain_grid

TIME = Literal["time"]
FREQ = Literal["freq"]


class _Wavelet(xr.DataArray):
    """Custom DataArray."""

    __slots__ = ()

    def plot(self, ax=None, cmap="bwr") -> plt.Figure:
        """Custom method."""
        return plot_wavelet_domain_grid(
            self.data, self.time.data, self.freq.data, ax=ax, cmap=cmap
        )


@dataclass
class TimeAxis:
    data: Data[TIME, int]
    long_name: Attr[str] = "Time"
    units: Attr[str] = "s"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


@dataclass
class FreqAxis:
    data: Data[FREQ, int]
    long_name: Attr[str] = "Frequency"
    units: Attr[str] = "Hz"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


@dataclass
class Wavelet(AsDataArray):
    data: Data[Tuple[FREQ, TIME], float]
    time: Coordof[TimeAxis] = 0
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Wavelet Amplitutde"

    __dataoptions__ = DataOptions(_Wavelet)


@dataclass
class TimeSeries(AsDataArray):
    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0
    name: Name[str] = "Time Series"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def sample_rate(self):
        return 1 / self.dt

    @property
    def duration(self):
        return self.time[-1] - self.time[0]

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def nyquist_frequency(self):
        return 1 / (2 * self.dt)


@dataclass
class FrequencySeries(AsDataArray):
    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Frequency Series"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @classmethod
    def from_time_series(cls, time_series: TimeSeries):
        freq = np.fft.rfftfreq(len(time_series), d=time_series.dt)
        data = np.fft.rfft(time_series)
        return cls(
            data=data,
            freq=FreqAxis(freq),
        )

    @property
    def df(self):
        return self.freq[1] - self.freq[0]

    @property
    def sample_rate(self):
        return self.df * len(self.freq)


def wavelet_dataset(
    wavelet_data: np.ndarray, time_grid=None, freq_grid=None, Nt=None, Nf=None
) -> Wavelet:
    """Create a dataset with wavelet coefficients.

    Parameters
    ----------
    wavelet : pywavelets.Wavelet object
        Wavelet to use.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with wavelet coefficients.
    """
    if Nt is None:
        Nt = wavelet_data.shape[1]
    if Nf is None:
        Nf = wavelet_data.shape[0]

    if time_grid is None:
        time_grid = np.arange(Nt)
    if freq_grid is None:
        freq_grid = np.arange(Nf)

    return Wavelet.new(wavelet_data, time=time_grid, freq=freq_grid)
