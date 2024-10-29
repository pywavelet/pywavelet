import matplotlib.pyplot as plt
from typing import Optional, Tuple
from .common import is_documented_by, xp, fmt_timerange
from .plotting import plot_wavelet_grid


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

    @is_documented_by(plot_wavelet_grid)
    def plot(self, ax=None, *args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        kwargs["time_grid"] = kwargs.get("time_grid", self.time)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq)
        return plot_wavelet_grid(wavelet_data=self.data, ax=ax, *args, **kwargs)

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
        return f"Wavelet(NfxNt={self.shape[0]}x{self.shape[1]}, {frange}Hz, {trange}s)"

    def __add__(self, other):
        """Element-wise addition of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(data=self.data + other.data, time=self.time, freq=self.freq)
        elif isinstance(other, float):
            return Wavelet(data=self.data + other, time=self.time, freq=self.freq)

    def __sub__(self, other):
        """Element-wise subtraction of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(data=self.data - other.data, time=self.time, freq=self.freq)
        elif isinstance(other, float):
            return Wavelet(data=self.data - other, time=self.time, freq=self.freq)

    def __mul__(self, other):
        """Element-wise multiplication of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(data=self.data * other.data, time=self.time, freq=self.freq)
        elif isinstance(other, float):
            return Wavelet(data=self.data * other, time=self.time, freq=self.freq)

    def __truediv__(self, other):
        """Element-wise division of two Wavelet objects."""
        if isinstance(other, Wavelet):
            return Wavelet(data=self.data / other.data, time=self.time, freq=self.freq)
        elif isinstance(other, float):
            return Wavelet(data=self.data / other, time=self.time, freq=self.freq)