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

    def plot(self, ax=None, cmap="bwr", log=False) -> plt.Figure:
        """Custom method."""
        return plot_wavelet_domain_grid(
            self.data,
            self.time.data,
            self.freq.data,
            ax=ax,
            cmap=cmap,
            log=log,
        )

    @property
    def Nt(self):
        return len(self.time)

    @property
    def Nf(self):
        return len(self.freq)

    @property
    def delta_t(self):
        return 1 / self.Nt

    @property
    def delta_f(self):
        return 1 / self.Nf


@dataclass
class TimeAxis:
    data: Data[TIME, float]
    long_name: Attr[str] = "Time"
    units: Attr[str] = "s"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


@dataclass
class FreqAxis:
    data: Data[FREQ, float]
    long_name: Attr[str] = "Frequency"
    units: Attr[str] = "Hz"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


@dataclass
class Wavelet(AsDataArray):
    data: Data[Tuple[FREQ, TIME], float]
    time: Coordof[TimeAxis] = 0.0
    freq: Coordof[FreqAxis] = 0.0
    name: Name[str] = "Wavelet Amplitutde"

    __dataoptions__ = DataOptions(_Wavelet)


@dataclass
class TimeSeries(AsDataArray):
    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0.0
    name: Name[str] = "Time Series"

    def plot(self, ax=None) -> plt.Figure:
        """Custom method."""
        if ax == None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.data)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

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

    @classmethod
    def from_frequency_series(cls, frequency_series: "FrequencySeries"):
        data = np.fft.irfft(frequency_series.data)
        dt = 1 / frequency_series.sample_rate
        return cls(
            data=data,
            time=TimeAxis(np.arange(len(data)) * dt),
        )


@dataclass
class FrequencySeries(AsDataArray):
    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Frequency Series"

    def plot(self, ax=None) -> plt.Figure:
        """Custom method."""
        if ax == None:
            fig, ax = plt.subplots()
        ax.loglog(self.freq, self.data)
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Amplitude")

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

    return Wavelet.new(wavelet_data.T, time=time_grid, freq=freq_grid)
