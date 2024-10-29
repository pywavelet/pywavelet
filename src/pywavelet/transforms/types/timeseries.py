import matplotlib.pyplot as plt
from typing import Tuple
from scipy.signal.spectral import spectrogram

from .common import is_documented_by, xp, rfft, rfftfreq, fmt_timerange, fmt_time
from .plotting import plot_timeseries, plot_spectrogram

__all__ = ["TimeSeries"]


class TimeSeries:
    """
    A class to represent a time series, with methods for plotting and converting
    the series to a frequency-domain representation.

    Attributes
    ----------
    data : xp.ndarray
        Time domain data.
    time : xp.ndarray
        Array of corresponding time points.
    """

    def __init__(self, data: xp.ndarray, time: xp.ndarray):
        """
        Initialize the TimeSeries with data and time arrays.

        Parameters
        ----------
        data : xp.ndarray
            Time domain data.
        time : xp.ndarray
            Array of corresponding time points. Must be the same length as `data`.

        Raises
        ------
        ValueError
            If `data` and `time` do not have the same length.
        """
        if len(data) != len(time):
            raise ValueError("data and time must have the same length")
        self.data = data
        self.time = time

    @is_documented_by(plot_timeseries)
    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_timeseries(self.data, self.time, ax=ax, **kwargs)

    @is_documented_by(plot_spectrogram)
    def plot_spectrogram(
            self, ax=None, spec_kwargs={}, plot_kwargs={}
    ) -> Tuple[plt.Figure, plt.Axes]:
        return plot_spectrogram(
            self.data,
            self.fs,
            ax=ax,
            spec_kwargs=spec_kwargs,
            plot_kwargs=plot_kwargs,
        )

    def __len__(self):
        """Return the number of data points in the time series."""
        return len(self.data)

    def __getitem__(self, item):
        """Return the data point at the specified index."""
        return self.data[item]

    @property
    def sample_rate(self) -> float:
        """
        Return the sample rate (fs).

        The sample rate is the inverse of the time resolution (Δt).
        """
        return float(xp.round(1.0 / self.dt, decimals=14))

    @property
    def fs(self) -> float:
        """Return the sample rate (fs)."""
        return self.sample_rate

    @property
    def duration(self) -> float:
        """Return the duration of the time series in seconds."""
        return len(self) / self.fs

    @property
    def dt(self) -> float:
        """Return the time resolution (Δt)."""
        return float(self.time[1] - self.time[0])

    @property
    def nyquist_frequency(self) -> float:
        """Return the Nyquist frequency (fs/2)."""
        return self.fs / 2

    @property
    def t0(self) -> float:
        """Return the initial time point in the series."""
        return float(self.time[0])

    @property
    def tend(self) -> float:
        """Return the final time point in the series."""
        return float(self.time[-1]) + self.dt

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the data array."""
        return self.data.shape

    @property
    def ND(self) -> int:
        """Return the number of data points in the time series."""
        return len(self)

    def __repr__(self) -> str:
        """Return a string representation of the TimeSeries."""
        trange = fmt_timerange((self.t0, self.tend))
        T = " ".join(fmt_time(self.duration, units=True))
        return f"TimeSeries(n={len(self)}, trange={trange}, T={T}, fs={self.fs:.2f} Hz)"

    def to_frequencyseries(self) -> 'FrequencySeries':
        """
        Convert the time series to a frequency series using the one-sided FFT.

        Returns
        -------
        FrequencySeries
            The frequency-domain representation of the time series.
        """
        freq = rfftfreq(len(self), d=self.dt)
        data = rfft(self.data)

        from .frequencyseries import FrequencySeries  # Avoid circular import
        return FrequencySeries(data, freq, t0=self.t0)

    def __add__(self, other: 'TimeSeries') -> 'TimeSeries':
        """Add two TimeSeries objects together."""
        if self.shape != other.shape:
            raise ValueError("TimeSeries objects must have the same shape to add them together")
        return TimeSeries(self.data + other.data, self.time)

    def __sub__(self, other: 'TimeSeries') -> 'TimeSeries':
        """Subtract one TimeSeries object from another."""
        if self.shape != other.shape:
            raise ValueError("TimeSeries objects must have the same shape to subtract them")
        return TimeSeries(self.data - other.data, self.time)
