from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Name

from .common import TIME, TimeAxis, _len_check, is_documented_by
from .plotting import plot_spectrogram, plot_timeseries

__all__ = ["TimeSeries"]


@dataclass
class TimeSeries(AsDataArray):
    """
    Represents a time series with associated data and methods for analysis and visualization.
    """

    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0.0
    name: Name[str] = "Time Series"

    def __post_init__(self):
        """Validate the data length after initialization."""
        _len_check(self.data)

    @is_documented_by(plot_timeseries)
    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the time series."""
        return plot_timeseries(self.data, self.time, ax=ax, **kwargs)

    @is_documented_by(plot_spectrogram)
    def plot_spectrogram(
        self,
        ax=None,
        spec_kwargs={},
        plot_kwargs={},
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the spectrogram of the time series."""
        return plot_spectrogram(
            self.data,
            self.fs,
            ax=ax,
            spec_kwargs=spec_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def __len__(self) -> int:
        """Return the length of the time series."""
        return len(self.data)

    def __getitem__(self, item):
        """Allow indexing of the time series data."""
        return self.data[item]

    @property
    def sample_rate(self) -> float:
        """Calculate and return the sample rate."""
        return np.round(1.0 / self.dt, decimals=14)

    @property
    def fs(self) -> float:
        """Alias for sample_rate."""
        return self.sample_rate

    @property
    def duration(self) -> float:
        """Calculate and return the duration of the time series."""
        return len(self) / self.fs

    @property
    def dt(self) -> float:
        """Calculate and return the time step."""
        return self.time[1] - self.time[0]

    @property
    def nyquist_frequency(self) -> float:
        """Calculate and return the Nyquist frequency."""
        return self.fs / 2

    @property
    def t0(self) -> float:
        """Return the start time of the series."""
        return self.time[0]

    @property
    def tend(self) -> float:
        """Return the end time of the series."""
        return self.time[-1]

    def __sub__(self, other: "TimeSeries") -> "TimeSeries":
        """Subtract two TimeSeries objects."""
        return TimeSeries(data=self.data - other.data, time=self.time)

    def __repr__(self) -> str:
        """Return a string representation of the TimeSeries."""
        return (
            f"TimeSeries(n={len(self)}, "
            f"trange=[{self.t0:.2f}, {self.tend:.2f}] s, "
            f"T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"
        )
