from typing import List

import numpy as np
from scipy.signal import chirp

from pywavelet.types import TimeSeries

__all__ = [
    "generate_chirp_time_domain_signal",
    "generate_sine_time_domain_signal",
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


def generate_sine_time_domain_signal(ts, f_true=10.0) -> TimeSeries:
    h_signal = np.sin(2 * np.pi * f_true * ts)
    return TimeSeries(h_signal, time=ts)
