import matplotlib.pyplot as plt
from typing import Tuple

from .common import is_documented_by, xp, irfft
from .plotting import plot_freqseries, plot_periodogram

__all__ = ["FrequencySeries"]

class FrequencySeries:
    """
    A class to represent a one-sided frequency series, with various methods for
    plotting and converting the series to a time-domain representation.

    Attributes
    ----------
    data : xp.ndarray
        Frequency domain data.
    freq : xp.ndarray
        Corresponding frequencies (must be non-negative).
    """

    def __init__(self, data: xp.ndarray, freq: xp.ndarray):
        """
        Initialize the FrequencySeries with data and frequencies.

        Parameters
        ----------
        data : xp.ndarray
            Frequency domain data.
        freq : xp.ndarray
            Array of frequencies. Must be non-negative.

        Raises
        ------
        ValueError
            If any frequency is negative or if `data` and `freq` do not have the same length.
        """
        if xp.any(freq < 0):
            raise ValueError("FrequencySeries must be one-sided (only non-negative frequencies)")
        if len(data) != len(freq):
            raise ValueError(f"data and freq must have the same length ({len(data)} != {len(freq)})")
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
        """Return the length of the frequency series."""
        return len(self.data)

    def __getitem__(self, item):
        """Return the data point at the specified index."""
        return self.data[item]

    @property
    def df(self) -> float:
        """Return the frequency resolution (Δf)."""
        return float(self.freq[1] - self.freq[0])

    @property
    def dt(self) -> float:
        """Return the time resolution (Δt)."""
        return 1 / self.fs

    @property
    def sample_rate(self) -> float:
        """Return the sample rate (fs)."""
        return 2 * float(self.freq[-1])

    @property
    def fs(self) -> float:
        """Return the sample rate (fs)."""
        return self.sample_rate

    @property
    def nyquist_frequency(self) -> float:
        """Return the Nyquist frequency (fs/2)."""
        return self.sample_rate / 2

    @property
    def duration(self) -> float:
        """Return the duration of the time domain signal."""
        return (2 * (len(self) - 1)) / self.fs

    @property
    def minimum_frequency(self) -> float:
        """Return the minimum frequency in the frequency series."""
        return float(xp.abs(self.freq).min())

    @property
    def maximum_frequency(self) -> float:
        """Return the maximum frequency in the frequency series."""
        return float(xp.abs(self.freq).max())

    @property
    def range(self) -> Tuple[float, float]:
        """Return the frequency range (minimum and maximum frequencies)."""
        return (self.minimum_frequency, self.maximum_frequency)

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the data array."""
        return self.data.shape

    @property
    def ND(self) -> int:
        """Return the number of data points in the time domain signal."""
        return 2 * (len(self) - 1)

    def __repr__(self) -> str:
        """Return a string representation of the FrequencySeries."""
        return f"FrequencySeries(n={len(self)}, frange=[{self.range[0]:.2f}, {self.range[1]:.2f}] Hz, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"

    def to_timeseries(self) -> "TimeSeries":
        """
        Convert the frequency series to a time series using inverse Fourier transform.

        Returns
        -------
        TimeSeries
            The corresponding time domain signal.
        """
        # Perform the inverse FFT
        time_data = irfft(self.data, n=2 * (len(self) - 1))

        # Calculate the time array
        dt = 1 / (2 * self.nyquist_frequency)
        time = xp.arange(len(time_data)) * dt

        # Create and return a TimeSeries object
        from .timeseries import TimeSeries
        return TimeSeries(time_data, time)
