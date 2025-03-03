from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .common import fmt_timerange, is_documented_by, xp
from .plotting import plot_wavelet_grid, plot_wavelet_trend
from .wavelet_bins import compute_bins


class Wavelet:
    """
    A class to represent a wavelet transform result, with methods for plotting and accessing key properties like duration, sample rate, and grid size.

    Attributes
    ----------
    data : xp.ndarray
        2D array representing the wavelet coefficients (frequency x time).
    time : xp.ndarray
        Array of time points.
    freq : xp.ndarray
        Array of corresponding frequency points.
    """

    def __init__(
        self,
        data: xp.ndarray,
        time: xp.ndarray,
        freq: xp.ndarray,
    ):
        """
        Initialize the Wavelet object with data, time, and frequency arrays.

        Parameters
        ----------
        data : xp.ndarray
            2D array representing the wavelet coefficients (frequency x time).
        time : xp.ndarray
            Array of time points.
        freq : xp.ndarray
            Array of corresponding frequency points.

        Raises
        ------
        AssertionError
            If the length of the time array does not match the number of time bins in `data`.
            If the length of the frequency array does not match the number of frequency bins in `data`.
        """
        nf, nt = data.shape
        assert len(time) == nt, f"len(time)={len(time)} != nt={nt}"
        assert len(freq) == nf, f"len(freq)={len(freq)} != nf={nf}"

        self.data = data
        self.time = time
        self.freq = freq

    @classmethod
    def zeros_from_grid(cls, time: xp.ndarray, freq: xp.ndarray) -> "Wavelet":
        """
        Create a Wavelet object filled with zeros.

        Parameters
        ----------
        time: xp.ndarray
        freq: xp.ndarray

        Returns
        -------
        Wavelet
            A Wavelet object with zero-filled data array.
        """
        Nf, Nt = len(freq), len(time)
        return cls(data=xp.zeros((Nf, Nt)), time=time, freq=freq)

    @classmethod
    def zeros(cls, Nf: int, Nt: int, T: float) -> "Wavelet":
        """
        Create a Wavelet object filled with zeros.

        Parameters
        ----------
        Nf : int
            Number of frequency bins.
        Nt : int
            Number of time bins.

        Returns
        -------
        Wavelet
            A Wavelet object with zero-filled data array.
        """
        return cls.zeros_from_grid(*compute_bins(Nf, Nt, T))

    @is_documented_by(plot_wavelet_grid)
    def plot(self, ax=None, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        kwargs["time_grid"] = kwargs.get("time_grid", self.time)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq)
        return plot_wavelet_grid(
            wavelet_data=self.data, ax=ax, *args, **kwargs
        )

    @is_documented_by(plot_wavelet_trend)
    def plot_trend(
        self, ax=None, *args, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        kwargs["time_grid"] = kwargs.get("time_grid", self.time)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq)
        return plot_wavelet_trend(
            wavelet_data=self.data, ax=ax, *args, **kwargs
        )

    @property
    def Nt(self) -> int:
        """
        Number of time bins.

        Returns
        -------
        int
            Length of the time array.
        """
        return len(self.time)

    @property
    def Nf(self) -> int:
        """
        Number of frequency bins.

        Returns
        -------
        int
            Length of the frequency array.
        """
        return len(self.freq)

    @property
    def ND(self) -> int:
        """
        Total number of data points in the wavelet grid.

        Returns
        -------
        int
            The product of `Nt` and `Nf`.
        """
        return self.Nt * self.Nf

    @property
    def delta_T(self) -> float:
        """
        Time resolution (ΔT) of the wavelet grid.

        Returns
        -------
        float
            Difference between consecutive time points.
        """
        return self.time[1] - self.time[0]

    @property
    def delta_F(self) -> float:
        """
        Frequency resolution (ΔF) of the wavelet grid.

        Returns
        -------
        float
            Inverse of twice the time resolution.
        """
        return 1 / (2 * self.delta_T)

    @property
    def duration(self) -> float:
        """
        Duration of the wavelet grid.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return float(self.Nt * self.delta_T)

    @property
    def delta_t(self) -> float:
        """
        Time resolution of the wavelet grid, normalized by the total number of data points.

        Returns
        -------
        float
            Time resolution per data point.
        """
        return float(self.duration / self.ND)

    @property
    def delta_f(self) -> float:
        """
        Frequency resolution of the wavelet grid, normalized by the total number of data points.

        Returns
        -------
        float
            Frequency resolution per data point.
        """
        return 1 / (2 * self.delta_t)

    @property
    def t0(self) -> float:
        """
        Initial time point of the wavelet grid.

        Returns
        -------
        float
            First time point in the time array.
        """
        return float(self.time[0])

    @property
    def tend(self) -> float:
        """
        Final time point of the wavelet grid.

        Returns
        -------
        float
            Last time point in the time array.
        """
        return float(self.time[-1])

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the wavelet grid.

        Returns
        -------
        Tuple[int, int]
            Tuple representing the shape of the data array (Nf, Nt).
        """
        return self.data.shape

    @property
    def sample_rate(self) -> float:
        """
        Sample rate of the wavelet grid.

        Returns
        -------
        float
            Sample rate calculated as the inverse of the time resolution.
        """
        return 1 / self.delta_t

    @property
    def fs(self) -> float:
        """
        Sample rate (fs) of the wavelet grid.

        Returns
        -------
        float
            The sample rate.
        """
        return self.sample_rate

    @property
    def nyquist_frequency(self) -> float:
        """
        Nyquist frequency of the wavelet grid.

        Returns
        -------
        float
            Nyquist frequency, which is half of the sample rate.
        """
        return self.sample_rate / 2

    def to_timeseries(self, nx: float = 4.0, mult: int = 32) -> "TimeSeries":
        """
        Convert the wavelet grid to a time-domain signal.

        Returns
        -------
        TimeSeries
            A `TimeSeries` object representing the time-domain signal.
        """
        from ..transforms import from_wavelet_to_time

        return from_wavelet_to_time(self, dt=self.delta_t, nx=nx, mult=mult)

    def to_frequencyseries(self, nx: float = 4.0) -> "FrequencySeries":
        """
        Convert the wavelet grid to a frequency-domain signal.

        Returns
        -------
        FrequencySeries
            A `FrequencySeries` object representing the frequency-domain signal.
        """
        from ..transforms import from_wavelet_to_freq

        return from_wavelet_to_freq(self, dt=self.delta_t, nx=nx)

    def __repr__(self) -> str:
        """
        Return a string representation of the Wavelet object.

        Returns
        -------
        str
            String containing information about the shape of the wavelet grid.
        """

        frange = ",".join([f"{f:.2e}" for f in (self.freq[0], self.freq[-1])])
        trange = fmt_timerange((self.t0, self.tend))
        Nfpow2 = int(xp.log2(self.shape[0]))
        Ntpow2 = int(xp.log2(self.shape[1]))
        shapef = f"NfxNf=[2^{Nfpow2}, 2^{Ntpow2}]"
        return f"Wavelet({shapef}, [{frange}]Hz, {trange})"

    def __add__(self, other):
        """Element-wise addition of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(
                data=self.data + other.data, time=self.time, freq=self.freq
            )
        elif isinstance(other, float):
            return Wavelet(
                data=self.data + other, time=self.time, freq=self.freq
            )

    def __sub__(self, other):
        """Element-wise subtraction of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(
                data=self.data - other.data, time=self.time, freq=self.freq
            )
        elif isinstance(other, float):
            return Wavelet(
                data=self.data - other, time=self.time, freq=self.freq
            )

    def __mul__(self, other):
        """Element-wise multiplication of two Wavelet objects."""
        if isinstance(other, WaveletMask):
            data = self.data.copy()
            data[~other.mask] = np.nan
            return Wavelet(data=data, time=self.time, freq=self.freq)
        elif isinstance(other, float):
            return Wavelet(
                data=self.data * other, time=self.time, freq=self.freq
            )
        elif isinstance(other, WaveletMask):
            return Wavelet(
                data=self.data * other.data, time=self.time, freq=self.freq
            )

    def __truediv__(self, other):
        """Element-wise division of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(
                data=self.data / other.data, time=self.time, freq=self.freq
            )
        elif isinstance(other, float):
            return Wavelet(
                data=self.data / other, time=self.time, freq=self.freq
            )

    def __eq__(self, other: "Wavelet") -> bool:
        """Element-wise comparison of two Wavelet objects."""
        data_all_same = xp.isclose(xp.nansum(self.data - other.data), 0)
        time_same = (self.time == other.time).all()
        freq_same = (self.freq == other.freq).all()
        return data_all_same and time_same and freq_same

    def noise_weighted_inner_product(
        self, other: "Wavelet", psd: "Wavelet"
    ) -> float:
        """
        Compute the noise-weighted inner product of two wavelet grids given a PSD.

        Parameters
        ----------
        other : Wavelet
            A `Wavelet` object representing the other wavelet grid.
        psd : Wavelet
            A `Wavelet` object representing the power spectral density.

        Returns
        -------
        float
            The noise-weighted inner product.
        """
        from ..utils import noise_weighted_inner_product

        return noise_weighted_inner_product(self, other, psd)

    def matched_filter_snr(self, template: "Wavelet", psd: "Wavelet") -> float:
        """
        Compute the matched filter SNR of the wavelet grid given a template.

        Parameters
        ----------
        template : Wavelet
            A `Wavelet` object representing the template.

        Returns
        -------
        float
            The matched filter signal-to-noise ratio.
        """
        mf = self.noise_weighted_inner_product(template, psd)
        return mf / self.optimal_snr(psd)

    def optimal_snr(self, psd: "Wavelet") -> float:
        """
        Compute the optimal SNR of the wavelet grid given a PSD.

        Parameters
        ----------
        psd : Wavelet
            A `Wavelet` object representing the power spectral density.

        Returns
        -------
        float
            The optimal signal-to-noise ratio.
        """
        return xp.sqrt(self.noise_weighted_inner_product(self, psd))

    def __copy__(self):
        return Wavelet(
            data=self.data.copy(), time=self.time.copy(), freq=self.freq.copy()
        )

    def copy(self):
        return self.__copy__()


class WaveletMask(Wavelet):
    @property
    def mask(self):
        return self.data

    def __repr__(self):
        rpr = super().__repr__()
        rpr = rpr.replace("Wavelet", "WaveletMask")
        return rpr

    @classmethod
    def from_restrictions(
        cls,
        time_grid: xp.ndarray,
        freq_grid: xp.ndarray,
        frange: List[float],
        tgaps: List[Tuple[float, float]] = [],
    ):
        """
        Create a WaveletMask object from restrictions on time and frequency.

        Parameters
        ----------
        time_grid : xp.ndarray
            Array of time points.
        freq_grid : xp.ndarray
            Array of corresponding frequency points.
        frange : List[float]
            Frequency range to include.
        tgaps : List[Tuple[float, float]]
            List of time gaps to exclude.

        Returns
        -------
        WaveletMask
            A WaveletMask object with the specified restrictions.
        """
        self = cls.zeros_from_grid(time_grid, freq_grid)
        self.data[(freq_grid >= frange[0]) & (freq_grid <= frange[1]), :] = (
            True
        )

        for tgap in tgaps:
            self.data[:, (time_grid >= tgap[0]) & (time_grid <= tgap[1])] = (
                False
            )
        self.data = self.data.astype(bool)
        return self
