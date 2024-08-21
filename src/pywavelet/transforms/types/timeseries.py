from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from xarray_dataclasses import AsDataArray, Coordof, Data, Name

from .common import TIME, TimeAxis, _len_check
from .plotting import plot_spectrogram, plot_timeseries

__all__ = ["TimeSeries"]


@dataclass
class TimeSeries(AsDataArray):
    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0.0
    name: Name[str] = "Time Series"

    def __post_init__(self):
        _len_check(self.data)

    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_timeseries(self.data, self.time, ax=ax, **kwargs)

    def plot_spectrogram(
        self,
        ax=None,
        spec_kwargs={},
        plot_kwargs={},
    ) -> Tuple[plt.Figure, plt.Axes]:
        return plot_spectrogram(
            self.data,
            self.fs,
            ax=ax,
            spec_kwargs=spec_kwargs,
            plot_kwargs=plot_kwargs,
        )

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
