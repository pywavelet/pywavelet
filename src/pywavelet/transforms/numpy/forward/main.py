from typing import Union

import numpy as np

from ....logger import logger
from ....types import FrequencySeries, TimeSeries, Wavelet
from ....types.wavelet_bins import _get_bins, _preprocess_bins
from ...phi_computer import phi_vec, phitilde_vec_norm
from .from_freq import transform_wavelet_freq_helper
from .from_time import transform_wavelet_time_helper

__all__ = ["from_time_to_wavelet", "from_freq_to_wavelet"]


def from_time_to_wavelet(
    timeseries: TimeSeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
    mult: int = 32,
) -> Wavelet:
    """
    Transform time-domain data to wavelet-domain data.

    This function performs a forward wavelet transform, converting a
    time-domain signal into a wavelet-domain representation.

    Parameters
    ----------
    timeseries : TimeSeries
        Input time-domain data, represented as a `TimeSeries` object.
    Nf : int, optional
        Number of frequency bins for the wavelet transform. Default is None.
    Nt : int, optional
        Number of time bins for the wavelet transform. Default is None.
    nx : float, optional
        Number of standard deviations for the `phi_vec`, controlling the
        width of the wavelets. Default is 4.0.
    mult : int, optional
        Number of time bins to use for the wavelet transform. Ensure `mult` is
        not larger than half the number of time bins (`Nt`). Default is 32.

    Returns
    -------
    Wavelet
        A `Wavelet` object representing the transformed wavelet-domain data.

    Warnings
    --------
    There can be significant leakage if `mult` is too small. The transform is
    only approximately exact if `mult = Nt / 2`.

    Notes
    -----
    The function warns when the `mult` value is too large, potentially leading
    to inaccurate results.
    """
    Nf, Nt = _preprocess_bins(timeseries, Nf, Nt)
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

    mult = min(mult, Nt // 2)  # Ensure mult is not larger than ND/2
    phi = phi_vec(Nf, d=nx, q=mult)
    wave = transform_wavelet_time_helper(timeseries.data, Nf, Nt, phi, mult).T
    return Wavelet(wave * np.sqrt(2), time=t_bins, freq=f_bins)


def from_freq_to_wavelet(
    freqseries: FrequencySeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
) -> Wavelet:
    """
    Transform frequency-domain data to wavelet-domain data.

    This function performs a forward wavelet transform, converting a
    frequency-domain signal into a wavelet-domain representation.

    Parameters
    ----------
    freqseries : FrequencySeries
        Input frequency-domain data, represented as a `FrequencySeries` object.
    Nf : int, optional
        Number of frequency bins for the wavelet transform. Default is None.
    Nt : int, optional
        Number of time bins for the wavelet transform. Default is None.
    nx : float, optional
        Number of standard deviations for the `phi_vec`, controlling the
        width of the wavelets. Default is 4.0.

    Returns
    -------
    Wavelet
        A `Wavelet` object representing the transformed wavelet-domain data.

    Notes
    -----
    The function normalizes the wavelet-domain data to ensure consistency
    during the transformation process.
    """
    Nf, Nt = _preprocess_bins(freqseries, Nf, Nt)
    t_bins, f_bins = _get_bins(freqseries, Nf, Nt)
    phif = phitilde_vec_norm(Nf, Nt, d=nx)
    wave = transform_wavelet_freq_helper(freqseries.data, Nf, Nt, phif)

    return Wavelet((2 / Nf) * wave.T * np.sqrt(2), time=t_bins, freq=f_bins)
