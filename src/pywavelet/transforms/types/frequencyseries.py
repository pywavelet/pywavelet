import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple, Union
from jax.numpy.fft import irfft

from .common import is_documented_by
from .plotting import plot_freqseries, plot_periodogram

__all__ = ["FrequencySeries"]

class FrequencySeries:
    def __init__(self, data: jnp.ndarray, freq: jnp.ndarray):
        if jnp.any(freq < 0):
            raise ValueError("FrequencySeries must be one-sided (only non-negative frequencies)")
        if len(data) != len(freq):
            raise ValueError("data and freq must have the same length")
        self.data = data
        self.freq = freq

    @is_documented_by(plot_freqseries)
    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_freqseries(
            self.data, self.freq, self.nyquist_frequency, ax=ax, **kwargs
        )

    @is_documented_by(plot_periodogram)
    def plot_periodogram(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_periodogram(
            self.data, self.freq, self.fs, ax=ax, **kwargs
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def df(self):
        return float(self.freq[1] - self.freq[0])

    @property
    def dt(self):
        return 1 / self.fs

    @property
    def sample_rate(self):
        return 2 * float(self.freq[-1])

    @property
    def fs(self):
        return self.sample_rate

    @property
    def nyquist_frequency(self):
        return float(self.freq[-1]) if self.freq[-1] <= self.sample_rate / 2 else self.sample_rate / 2

    @property
    def duration(self):
        return (len(self) - 1) / self.sample_rate

    @property
    def minimum_frequency(self):
        return float(self.freq[0])

    @property
    def maximum_frequency(self):
        return float(self.freq[-1])

    @property
    def range(self) -> Tuple[float, float]:
        return (self.minimum_frequency, self.maximum_frequency)

    def __repr__(self):
        return f"FrequencySeries(n={len(self)}, frange=[{self.range[0]:.2f}, {self.range[1]:.2f}] Hz, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"


    def __add__(self, other: Union['FrequencySeries', float, int]) -> 'FrequencySeries':
        if isinstance(other, (float, int)):
            return FrequencySeries(self.data + other, self.freq)
        elif isinstance(other, FrequencySeries):
            if not jnp.allclose(self.freq, other.freq):
                raise ValueError("Frequency grids must be the same for addition")
            return FrequencySeries(self.data + other.data, self.freq)
        else:
            return NotImplemented

    def __sub__(self, other: Union['FrequencySeries', float, int]) -> 'FrequencySeries':
        if isinstance(other, (float, int)):
            return FrequencySeries(self.data - other, self.freq)
        elif isinstance(other, FrequencySeries):
            if not jnp.allclose(self.freq, other.freq):
                raise ValueError("Frequency grids must be the same for subtraction")
            return FrequencySeries(self.data - other.data, self.freq)
        else:
            return NotImplemented

    def __mul__(self, other: Union['FrequencySeries', float, int]) -> 'FrequencySeries':
        if isinstance(other, (float, int)):
            return FrequencySeries(self.data * other, self.freq)
        elif isinstance(other, FrequencySeries):
            if not jnp.allclose(self.freq, other.freq):
                raise ValueError("Frequency grids must be the same for multiplication")
            return FrequencySeries(self.data * other.data, self.freq)
        else:
            return NotImplemented

    def __truediv__(self, other: Union['FrequencySeries', float, int]) -> 'FrequencySeries':
        if isinstance(other, (float, int)):
            return FrequencySeries(self.data / other, self.freq)
        elif isinstance(other, FrequencySeries):
            if not jnp.allclose(self.freq, other.freq):
                raise ValueError("Frequency grids must be the same for division")
            return FrequencySeries(self.data / other.data, self.freq)
        else:
            return NotImplemented

    def to_timeseries(self) -> "TimeSeries":
        """Convert frequency series to time series using inverse Fourier transform."""
        # Perform the inverse FFT
        time_data = irfft(self.data, n=2 * (len(self) - 1))

        # Calculate the time array
        dt = 1 / (2 * self.nyquist_frequency)
        time = jnp.arange(len(time_data)) * dt

        # Create and return a TimeSeries object
        from .timeseries import TimeSeries
        return TimeSeries(time_data, time)