import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from .common import is_documented_by
from .plotting import plot_wavelet_grid



class Wavelet:
    def __init__(
            self,
            data: jnp.ndarray,
            time: jnp.ndarray,
            freq: jnp.ndarray,
    ):
        self.data = data
        self.time = time
        self.freq = freq


    @is_documented_by(plot_wavelet_grid)
    def plot(self, ax=None, *args, **kwargs) -> plt.Figure:
        """Plot the wavelet grid."""
        from .plotting import plot_wavelet_grid  # Import here to avoid circular imports
        kwargs["time_grid"] = kwargs.get("time_grid", self.time)
        kwargs["freq_grid"] = kwargs.get("freq_grid", self.freq)
        return plot_wavelet_grid(wavelet_data=self.data, ax=ax, *args, **kwargs)

    @property
    def Nt(self) -> int:
        """Number of time bins."""
        return len(self.time)

    @property
    def Nf(self) -> int:
        """Number of frequency bins."""
        return len(self.freq)

    @property
    def ND(self) -> int:
        return self.Nt * self.Nf

    @property
    def delta_T(self):
        return self.time[1] - self.time[0]

    @property
    def delta_F(self):
        return 1 / (2 * self.delta_T)

    @property
    def duration(self) -> float:
        return float(self.Nt * self.delta_T)

    @property
    def delta_t(self) -> float:
        return float(self.duration / self.ND)

    @property
    def delta_f(self) -> float:
        return 1 / (2 * self.delta_t)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the wavelet grid. (Nf, Nt)"""
        return self.data.shape

    @property
    def sample_rate(self) -> float:
        return 1 / self.delta_t

    @property
    def fs(self):
        return self.sample_rate

    @property
    def nyquist_frequency(self) -> float:
        return self.sample_rate / 2

    def __repr__(self):
        return f"Wavelet(NfxNt={self.shape[0]}x{self.shape[1]})"

    @classmethod
    def from_data(
        cls,
        data: jnp.ndarray,
        time_grid: Optional[jnp.ndarray] = None,
        freq_grid: Optional[jnp.ndarray] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> "Wavelet":
        """Create a Wavelet instance from data."""
        if time_grid is None:
            time_grid = jnp.arange(data.shape[1])
        if freq_grid is None:
            freq_grid = jnp.arange(data.shape[0])

        w = cls(data, time_grid, freq_grid)

        if freq_range is not None:
            freq_mask = (w.freq >= freq_range[0]) & (w.freq <= freq_range[1])
            w.data = w.data[freq_mask]
            w.freq = w.freq[freq_mask]

        if time_range is not None:
            time_mask = (w.time >= time_range[0]) & (w.time <= time_range[1])
            w.data = w.data[:, time_mask]
            w.time = w.time[time_mask]

        return w
