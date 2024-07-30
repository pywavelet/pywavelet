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

from .common import TIME, TimeAxis, _len_check

__all__ = ["TimeSeries"]


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
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
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
