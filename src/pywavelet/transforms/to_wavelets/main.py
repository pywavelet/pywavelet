from typing import Union

import numpy as np

from ...logger import logger
from ..phi_computer import phi_vec, phitilde_vec_norm
from ..types import FrequencySeries, TimeSeries, Wavelet
from .transform_freq_funcs import transform_wavelet_freq_helper
from .transform_time_funcs import transform_wavelet_time_helper
from .wavelet_bins import _get_bins, _preprocess_bins


def from_time_to_wavelet(
    timeseries: TimeSeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
    mult: int = 32,
    **kwargs,
) -> Wavelet:
    """From time domain to wavelet domain

    Warning: there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2

    Parameters
    ----------
    timeseries : array_like
        Time domain freqseries
    Nf : int
        Number of frequency bins
    Nt : int
        Number of time bins
    nx : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.
    mult : int, optional
        Number of time bins to use for the wavelet transform, by default 32
    """
    Nf, Nt = _preprocess_bins(timeseries, Nf, Nt)
    dt = timeseries.dt
    t_bins, f_bins = _get_bins(timeseries, Nf, Nt)

    ND = Nf * Nt

    if len(timeseries) != ND:
        logger.warning(
            f"len(freqseries)={len(timeseries)} != Nf*Nt={ND}. Truncating to freqseries[:{ND}]"
        )
        timeseries = timeseries[:ND]
    if mult > Nt / 2:
        logger.warning(
            f"mult={mult} is too large for Nt={Nt}. This may lead to bogus results."
        )

    mult = min(mult, Nt // 2)  # make sure K isn't bigger than ND
    phi = phi_vec(Nf, dt=dt, d=nx, q=mult)
    wave = transform_wavelet_time_helper(timeseries.data, Nf, Nt, phi, mult)

    wave = wave * np.sqrt(2)

    return Wavelet.from_data(
        wave, time_grid=t_bins, freq_grid=f_bins, **kwargs
    )


def from_freq_to_wavelet(
    freqseries: FrequencySeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
    **kwargs,
) -> Wavelet:
    """do the wavelet transform using the fast wavelet domain transform"""
    Nf, Nt = _preprocess_bins(freqseries, Nf, Nt)
    t_bins, f_bins = _get_bins(freqseries, Nf, Nt)
    dt = freqseries.dt
    phif = phitilde_vec_norm(Nf, Nt, dt=dt, d=nx)
    wave = (2 / Nf) * transform_wavelet_freq_helper(
        freqseries.data, Nf, Nt, phif
    )
    wave = wave * 2 ** (1 / 2)
    return Wavelet.from_data(
        wave, time_grid=t_bins, freq_grid=f_bins, **kwargs
    )
