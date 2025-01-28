from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from ..backend import rfft, rfftfreq, xp
from ..logger import logger
from .common import fmt_pow2, fmt_time, fmt_timerange, is_documented_by
from .plotting import plot_spectrogram, plot_timeseries

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
        n = fmt_pow2(len(self))
        return (
            f"TimeSeries(n={n}, trange={trange}, T={T}, fs={self.fs:.2f} Hz)"
        )

    def to_frequencyseries(self) -> "FrequencySeries":
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

    def to_wavelet(
        self,
        Nf: Union[int, None] = None,
        Nt: Union[int, None] = None,
        nx: Optional[float] = 4.0,
    ) -> "Wavelet":
        """
        Convert the time series to a wavelet representation.

        Parameters
        ----------
        Nf : int
            Number of frequency bins for the wavelet transform.
        Nt : int
            Number of time bins for the wavelet transform.
        nx : float, optional
            Number of standard deviations for the `phi_vec`, controlling the
            width of the wavelets. Default is 4.0.

        Returns
        -------
        Wavelet
            The wavelet-domain representation of the time series.
        """
        hf = self.to_frequencyseries()
        return hf.to_wavelet(Nf, Nt, nx=nx)

    def __add__(self, other: "TimeSeries") -> "TimeSeries":
        """Add two TimeSeries objects together."""
        if self.shape != other.shape:
            raise ValueError(
                "TimeSeries objects must have the same shape to add them together"
            )
        return TimeSeries(self.data + other.data, self.time)

    def __sub__(self, other: "TimeSeries") -> "TimeSeries":
        """Subtract one TimeSeries object from another."""
        if self.shape != other.shape:
            raise ValueError(
                "TimeSeries objects must have the same shape to subtract them"
            )
        return TimeSeries(self.data - other.data, self.time)

    def __eq__(self, other: "TimeSeries") -> bool:
        """Check if two TimeSeries objects are equal."""
        shape_same = self.shape == other.shape
        range_same = self.t0 == other.t0 and self.tend == other.tend
        time_same = xp.allclose(self.time, other.time)
        data_same = xp.allclose(self.data, other.data)
        return shape_same and range_same and data_same and time_same

    def __mul__(self, other: float) -> "TimeSeries":
        """Multiply a TimeSeries object by a scalar."""
        return TimeSeries(self.data * other, self.time)

    def zero_pad_to_power_of_2(
        self, tukey_window_alpha: float = 0.0
    ) -> "TimeSeries":
        """Zero pad the time series to make the length a power of two (useful to speed up FFTs, O(NlogN) versus O(N^2)).

        Parameters
        ----------
        tukey_window_alpha : float, optional
            Alpha parameter for the Tukey window. Default is 0.0.
            (prevents spectral leakage when padding the data)

        Returns
        -------
        TimeSeries
            A new TimeSeries object with the data zero-padded to a power of two.
        """
        N, dt, t0 = self.ND, self.dt, self.t0
        pow_2 = xp.ceil(xp.log2(N))
        n_pad = int((2**pow_2) - N)
        new_N = N + n_pad
        if n_pad > 0:
            logger.warning(
                f"Padding the data to a power of two. "
                f"{N:,} (2**{xp.log2(N):.2f}) -> {new_N:,} (2**{pow_2}). "
            )
        window = tukey(N, alpha=tukey_window_alpha)
        data = self.data * window
        data = xp.pad(data, (0, n_pad), "constant")
        time = xp.arange(0, len(data) * dt, dt) + t0
        return TimeSeries(data, time)

    def highpass_filter(
        self,
        fmin: float,
        tukey_window_alpha: float = 0.0,
        bandpass_order: int = 4,
    ) -> "TimeSeries":
        """
        Filter the time series with a highpass bandpass filter.

        (we use sosfiltfilt instead of filtfilt for numerical stability)

        Note: filtfilt should be used if phase accuracy (zero-phase filtering) is critical for your analysis
        and if the filter order is low to moderate.


        Parameters
        ----------
        fmin : float
            Minimum frequency to pass through the filter.
        bandpass_order : int, optional
            Order of the bandpass filter. Default is 4.

        Returns
        -------
        TimeSeries
            A new TimeSeries object with the highpass filter applied.
        """

        if fmin <= 0 or fmin > self.nyquist_frequency:
            raise ValueError(
                f"Invalid fmin value: {fmin}. Must be in the range [0, {self.nyquist_frequency}]"
            )

        sos = butter(
            bandpass_order, Wn=fmin, btype="highpass", output="sos", fs=self.fs
        )
        data = self.data.copy()
        data = sosfiltfilt(sos, data)
        data = data * tukey(self.ND, alpha=tukey_window_alpha)
        return TimeSeries(data, self.time)

    def __copy__(self):
        return TimeSeries(self.data.copy(), self.time.copy())

    def copy(self):
        return self.__copy__()

    def __getitem__(self, key) -> "TimeSeries":
        if isinstance(key, slice):
            # Handle slicing
            return self.__handle_slice(key)
        else:
            # Handle regular indexing
            return TimeSeries(self.data[key], self.time[key])

    def __handle_slice(self, slice_obj) -> "TimeSeries":
        return TimeSeries(self.data[slice_obj], self.time[slice_obj])

    @classmethod
    def _EMPTY(cls, ND: int, dt: float) -> "TimeSeries":
        return cls(xp.zeros(ND), xp.arange(0, ND * dt, dt))
