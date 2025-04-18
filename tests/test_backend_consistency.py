from scipy.signal import chirp
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from typing import Tuple, Union

jax.config.update('jax_enable_x64', True)

from matplotlib.pyplot import GridSpec
from pywavelet.types import TimeSeries, FrequencySeries, Wavelet
from pywavelet.transforms.numpy import from_freq_to_wavelet, from_wavelet_to_freq
from pywavelet.transforms.jax import from_freq_to_wavelet as jax_from_freq_to_wavelet
from pywavelet.transforms.jax import from_wavelet_to_freq as jax_from_wavelet_to_freq
from dataclasses import dataclass

from pywavelet.types import Wavelet, FrequencySeries, TimeSeries
from pywavelet.types.wavelet_bins import compute_bins
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from utils.plotting import plot_roundtrip



class MonochromaticSignal:
    def __init__(self, f0: float, dt: float, A: float, Nt: int, Nf: int):
        self.f0 = f0
        self.dt = dt
        self.A = A
        self.Nt = Nt
        self.Nf = Nf
        self.ND = Nt * Nf
        self.T = Nt * Nf * dt

    def __repr__(self):
        return f"MonochromaticSignal(f0={self.f0}, dt={self.dt}, A={self.A})"

    @property
    def wavelet(self) -> Wavelet:
        if hasattr(self, "_wavelet"):
            return self._wavelet

        Nf, Nt, T = self.Nf, self.Nt, self.T
        f0, dt, A = self.f0, self.dt, self.A
        N = Nt * Nf

        t_bins, f_bins = compute_bins(Nf, Nt, T)
        wnm = np.zeros((Nt, Nf))
        m0 = int(f0 * N * dt)
        f0_bin_idx = int(2 * m0 / Nt)
        odd_t_indices = np.arange(Nt) % 2 != 0
        wnm[odd_t_indices, f0_bin_idx] = A * np.sqrt(2 * Nf)
        self._wavelet = Wavelet(wnm.T, t_bins, f_bins)
        return self._wavelet


    @property
    def frequencyseries(self) -> FrequencySeries:
        if hasattr(self, "_frequencyseries"):
            return self._frequencyseries
        ND = self.ND
        dt = self.dt
        t = np.arange(0, ND) * dt
        y = self.A * np.sin(2 * np.pi * self.f0 * t)
        ts = TimeSeries(data=y, time=t)
        self._frequencyseries = ts.to_frequencyseries()
        return self._frequencyseries



@dataclass
class RoundtripData:
    wavelet: Wavelet
    reconstructed: FrequencySeries




def plot(
        true: MonochromaticSignal,
        np_data: RoundtripData,
        jax_data: RoundtripData,
        plot_fn: str,
) -> None:
    # make a gridspec 3 rows 3 columns (last row is full width)
    fig,axes = plt.subplots(2,4,  figsize=(14, 7))

    plot_roundtrip(
        true.frequencyseries,
        true.wavelet,
        np_data.reconstructed,
        np_data.wavelet,
        axes=axes[0, :],
        labels=["Analytic", "Numpy"],
    )
    plot_roundtrip(
        true.frequencyseries,
        true.wavelet,
        jax_data.reconstructed,
        jax_data.wavelet,
        axes=axes[1, :],
        labels=["Analytic", "JAX"],
    )
    plt.tight_layout()
    plt.savefig(plot_fn)





def test_jax_and_numpy(plot_dir):
    # Sizes
    fs = 32
    dt = 1 / fs
    f0 = 8.0
    Nt, Nf = 2 ** 3, 2 ** 3

    signal = MonochromaticSignal(f0=f0, dt=dt, A=2, Nt=Nt, Nf=Nf)

    # using numpy
    h_wavelet = from_freq_to_wavelet(signal.frequencyseries, Nf=Nf, Nt=Nt)
    h_reconstructed = from_wavelet_to_freq(h_wavelet, dt=dt)
    np_data = RoundtripData(wavelet=h_wavelet, reconstructed=h_reconstructed)

    # using JAX
    jax_h_wavelet = jax_from_freq_to_wavelet(signal.frequencyseries, Nf=Nf, Nt=Nt)
    jax_h_reconstructed = jax_from_wavelet_to_freq(jax_h_wavelet, dt=dt)
    jax_data = RoundtripData(wavelet=jax_h_wavelet, reconstructed=jax_h_reconstructed)

    # Plotting
    plot(
        signal,
        np_data,
        jax_data,
        plot_fn=f"{plot_dir}/jax_vs_numpy_roundtrip.png",
    )
