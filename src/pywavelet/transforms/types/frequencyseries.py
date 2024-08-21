from dataclasses import dataclass, field
from typing import Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from xarray_dataclasses import AsDataArray, Coordof, Data, Name

from .common import FREQ, FreqAxis, is_documented_by
from .plotting import plot_freqseries, plot_periodogram

__all__ = ["FrequencySeries"]

# Type aliases
PlotResult: TypeAlias = Tuple[plt.Figure, plt.Axes]


@dataclass
class FrequencySeries(AsDataArray):
    """
    Represents a frequency series with associated data and methods for analysis and visualization.
    """

    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = field(default=0)
    name: Name[str] = "Frequency Series"

    def __post_init__(self):
        """
        Validate the frequency values after initialization.
        """
        if np.any(self.freq <= 0):
            raise ValueError("All frequency values must be greater than zero.")

    # Plotting methods
    @is_documented_by(plot_freqseries)
    def plot(self, ax=None, **kwargs) -> PlotResult:
        return plot_freqseries(
            self.data, self.freq, self.nyquist_frequency, ax=ax, **kwargs
        )

    @is_documented_by(plot_periodogram)
    def plot_periodogram(self, ax=None, **kwargs) -> PlotResult:
        return plot_periodogram(
            self.data, self.freq, self.nyquist_frequency, ax=ax, **kwargs
        )

    # Data access methods
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    # Frequency-related properties
    @property
    def df(self) -> float:
        """Frequency resolution."""
        return self.freq[1] - self.freq[0]

    @property
    def minimum_frequency(self) -> float:
        """Minimum frequency in the series."""
        return min(self.freq)

    @property
    def maximum_frequency(self) -> float:
        """Maximum frequency in the series."""
        return max(self.freq)

    @property
    def freq_range(self) -> Tuple[float, float]:
        """Frequency range as (min, max)."""
        return (self.minimum_frequency, self.maximum_frequency)

    @property
    def nyquist_frequency(self) -> float:
        """Nyquist frequency of the series."""
        return self.freq[-1]

    # Time-related properties
    @property
    def dt(self) -> float:
        """Time resolution."""
        return 1 / self.fs

    @property
    def duration(self) -> float:
        """Total duration of the time series."""
        return 2 * self.dt * (len(self) - 1)

    # Sampling-related properties
    @property
    def sample_rate(self) -> float:
        """Sampling rate of the series."""
        return self.nyquist_frequency * 2

    @property
    def fs(self) -> float:
        """Alias for sample_rate."""
        return self.sample_rate

    def __repr__(self) -> str:
        """String representation of the FrequencySeries."""
        return (
            f"FrequencySeries(n={len(self)}, "
            f"frange=[{self.minimum_frequency:.2f}, {self.maximum_frequency:.2f}] Hz, "
            f"T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"
        )
