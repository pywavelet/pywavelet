import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from .common import is_documented_by, xp
from .plotting import plot_wavelet_grid



class Wavelet:
    def __init__(
            self,
            data: xp.ndarray,
            time: xp.ndarray,
            freq: xp.ndarray,
    ):
        nf, nt = data.shape
        assert len(time) == nt, f"len(time)={len(time)} != nt={nt}"
        assert len(freq) == nf, f"len(freq)={len(freq)} != nf={nf}"

        self.data = data
        self.time = time
        self.freq = freq





    @is_documented_by(plot_wavelet_grid)
    def plot(self, ax=None, *args, **kwargs) -> plt.Figure:
        """Plot the wavelet grid."""
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