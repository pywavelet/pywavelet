from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import spectrogram
from xarray_dataclasses import (
    AsDataArray,
    Attr,
    Coordof,
    Data,
    DataOptions,
    Name,
)

from ..logger import logger
from ..plotting import plot_wavelet_grid

TIME = Literal["time"]
FREQ = Literal["freq"]


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
    def delta_t(self) -> float:
        return 1 / self.Nt

    @property
    def delta_f(self) -> float:
        return 1 / self.Nf

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
    def duration(self) -> float:
        return self.Nt * self.delta_t

    @property
    def nyquist_frequency(self) -> float:
        return self.sample_rate / 2


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
    name: Name[str] = "Wavelet Amplitude"

    __dataoptions__ = DataOptions(_Wavelet)


@dataclass
class TimeSeries(AsDataArray):
    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0.0
    name: Name[str] = "Time Series"

    def __post_init__(self):
        _len_check(self.data)

    def plot(self, ax=None, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Custom method."""
        if ax == None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.data, **kwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(left=self.time[0], right=self.time[-1])
        return ax.figure, ax

    def plot_spectrogram(
        self, ax=None, spec_kwargs={}, plot_kwargs={}, *args, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        f, t, Sxx = spectrogram(self.data, fs=self.fs, **spec_kwargs)
        if ax == None:
            fig, ax = plt.subplots()

        if "cmap" not in plot_kwargs:
            plot_kwargs["cmap"] = "Reds"

        cm = ax.pcolormesh(t, f, Sxx, shading="nearest", **plot_kwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_ylim(top=self.nyquist_frequency)
        cbar = plt.colorbar(cm, ax=ax)
        cbar.set_label("Spectrogram Amplitude")
        return ax.figure, ax

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def sample_rate(self) -> float:
        return np.round(1.0 / self.dt, decimals=14)

    @property
    def fs(self):
        return self.sample_rate

    @property
    def duration(self):
        return len(self) / self.fs

    @property
    def dt(self):
        return self.time[1] - self.time[0]

    @property
    def nyquist_frequency(self):
        return self.fs / 2

    @property
    def t0(self):
        return self.time[0]

    @property
    def tend(self):
        return self.time[-1]

    def __sub__(self, other):
        return TimeSeries(data=self.data - other.data, time=self.time)

    def __repr__(self):
        return f"TimeSeries(n={len(self)}, trange=[{self.time[0]:.2f}, {self.time[-1]:.2f}] s, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"


@dataclass
class FrequencySeries(AsDataArray):
    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Frequency Series"

    def plot(self, ax=None, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        if ax == None:
            fig, ax = plt.subplots()
        ax.plot(self.freq, self.data, **kwargs)
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-self.nyquist_frequency, self.nyquist_frequency)
        return ax.figure, ax

    def plot_periodogram(
        self, ax=None, *args, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        if ax == None:
            fig, ax = plt.subplots()

        ax.loglog(self.freq, np.abs(self.data) ** 2, **kwargs)
        flow = np.min(np.abs(self.freq))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_xlim(left=flow, right=self.nyquist_frequency)
        return ax.figure, ax

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def df(self):
        return self.freq[1] - self.freq[0]

    @property
    def dt(self):
        return 1 / self.fs

    @property
    def sample_rate(self):
        return self.nyquist_frequency * 2

    @property
    def fs(self):
        return self.sample_rate

    @property
    def nyquist_frequency(self):
        return self.freq[-1]

    @property
    def duration(self):
        return 2 * self.dt * (len(self) - 1)

    @property
    def freq_range(self) -> Tuple[float, float]:
        return (min(self.freq), max(self.freq))

    def __repr__(self):
        return f"FrequencySeries(n={len(self)}, frange=[{min(self.freq):.2f}, {max(self.freq):.2f}] Hz, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"


def wavelet_dataset(
    wavelet_data: np.ndarray,
    time_grid=None,
    freq_grid=None,
    freq_range=None,
    time_range=None,
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
    w = Wavelet.new(wavelet_data.T, time=time_grid, freq=freq_grid)

    if freq_range is not None:
        w = w.sel(freq=slice(*freq_range))
    if time_range is not None:
        w = w.sel(time=slice(*time_range))
    return w


def _len_check(d):
    if not np.log2(len(d)).is_integer():
        logger.warning(f"Data length {len(d)} is suggested to be a power of 2")
