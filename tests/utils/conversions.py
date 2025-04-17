from typing import Union

import jax.numpy as jnp
import numpy as np

from pywavelet.backend import cuda_available
from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

if cuda_available:
    import cupy as cp


def to_jax(
    d: Union[TimeSeries, FrequencySeries, Wavelet],
) -> Union[TimeSeries, FrequencySeries, Wavelet]:
    d.data = jnp.array(d.data)
    return d


def to_numpy(
    d: Union[TimeSeries, FrequencySeries, Wavelet],
) -> Union[TimeSeries, FrequencySeries, Wavelet]:
    if cuda_available:
        if isinstance(d.data, cp.ndarray):
            d.data = d.data.get()
    d.data = np.array(d.data)
    return d


def to_cupy(
    d: Union[TimeSeries, FrequencySeries, Wavelet],
) -> Union[TimeSeries, FrequencySeries, Wavelet]:
    d.data = cp.array(d.data)
    return d
