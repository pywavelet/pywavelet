from typing import List

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

from pywavelet.types import FrequencySeries, TimeSeries

__all__ = [
    "generate_chirp_time_domain_signal",
    "generate_sine_time_domain_signal",
    "generate_pure_f0",
]


def generate_chirp_time_domain_signal(
    t: np.ndarray, freq_range: List[float]
) -> TimeSeries:
    fs = 1 / (t[1] - t[0])
    nyquist = fs / 2
    fmax = max(freq_range)
    assert (
        fmax < nyquist
    ), f"f_max [{fmax:.2f} Hz] must be less than f_nyquist [{nyquist:2f} Hz]."

    y = chirp(
        t, f0=freq_range[0], f1=freq_range[1], t1=t[-1], method="quadratic"
    )
    return TimeSeries(data=y, time=t)


def generate_sine_time_domain_signal(ts, n, f_true=10):
    h_signal = np.sin(2 * np.pi * f_true * ts)
    window = tukey(n, 0.0)
    h_signal = __zero_pad(h_signal * window)
    return TimeSeries(h_signal, time=ts)


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def generate_pure_f0(
    f0=1,
    Nf=8,
    Nt=4,
    dt=0.1,
) -> FrequencySeries:
    N = Nf * Nt
    freq = np.fft.rfftfreq(N, dt)
    hf = np.zeros_like(freq, dtype=np.complex128)
    hf[np.argmin(np.abs(freq - f0))] = 1.0
    return FrequencySeries(data=hf, freq=freq)
