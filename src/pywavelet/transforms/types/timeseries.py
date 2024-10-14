import matplotlib.pyplot as plt
from typing import Tuple, Union
from scipy.signal.spectral import spectrogram

from .common import is_documented_by, xp, rfft, rfftfreq
from .plotting import plot_timeseries, plot_spectrogram

__all__ = ["TimeSeries"]

class TimeSeries:
    def __init__(self, data: xp.ndarray, time: xp.ndarray):
        if len(data) != len(time):
            raise ValueError("data and time must have the same length")
        self.data = data
        self.time = time

    @is_documented_by(plot_timeseries)
    def plot(self, ax=None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return plot_timeseries(self.data, self.time, ax=ax, **kwargs)

    @is_documented_by(plot_spectrogram)
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
        return float(xp.round(1.0 / self.dt, decimals=14))

    @property
    def fs(self):
        return self.sample_rate

    @property
    def duration(self):
        return len(self) / self.fs

    @property
    def dt(self):
        return float(self.time[1] - self.time[0])

    @property
    def nyquist_frequency(self):
        return self.fs / 2

    @property
    def t0(self):
        return float(self.time[0])

    @property
    def tend(self):
        return float(self.time[-1]) + self.dt

    @property
    def shape(self):
        return self.data.shape


    @property
    def ND(self):
        return len(self)

    def __repr__(self):
        return f"TimeSeries(n={len(self)}, trange=[{self.t0:.2f}, {self.tend:.2f}] s, T={self.duration:.2f}s, fs={self.fs:.2f} Hz)"

    def to_frequencyseries(self) -> 'FrequencySeries':
        """Convert time series to frequency series using the ONE SIDED FFT transform."""
        freq = rfftfreq(len(self), d=self.dt)
        data = rfft(self.data)
        # This is all the +ive frequencies
        from .frequencyseries import FrequencySeries  # Avoid circular import
        return FrequencySeries(data, freq)