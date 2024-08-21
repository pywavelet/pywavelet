from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Name

from .common import FREQ, FreqAxis
from .plotting import plot_freqseries, plot_periodogram

__all__ = ["FrequencySeries"]


@dataclass
class FrequencySeries(AsDataArray):
    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Frequency Series"

    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_freqseries(
            self.data, self.freq, self.nyquist_frequency, ax=ax, **kwargs
        )

    def plot_periodogram(
        self, ax=None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        return plot_periodogram(
            self.data, self.freq, self.nyquist_frequency, ax=ax, **kwargs
        )

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
    def minimum_frequency(self):
        return min(self.freq)

    @property
    def maximum_frequency(self):
        return max(self.freq)

    @property
    def freq_range(self) -> Tuple[float, float]:
        return (self.minimum_frequency, self.maximum_frequency)

    def __repr__(self):
        return f"FrequencySeries(n={len(self)}, frange=[{min(self.freq):.2f}, {max(self.freq):.2f}] Hz, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"
