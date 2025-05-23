from typing import Union

import cupy as cp

from .... import backend
from ....logger import logger
from ....types import FrequencySeries, TimeSeries, Wavelet
from ....types.wavelet_bins import _get_bins, _preprocess_bins
from ...phi_computer import phi_vec, phitilde_vec_norm
from .from_freq import transform_wavelet_freq_helper
from .from_time import transform_wavelet_time_helper


def from_time_to_wavelet(
    timeseries: TimeSeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
    mult: int = 32,
) -> Wavelet:
    """Transforms time-domain data to wavelet-domain data.

    Warning: there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2

    Parameters
    ----------
    timeseries : TimeSeries
        Time domain data
    Nf : int
        Number of frequency bins
    Nt : int
        Number of time bins
    nx : float, optional
        Number of standard deviations for the phi_vec, by default 4.
    mult : int, optional
        Number of time bins to use for the wavelet transform, by default 32
    **kwargs:
        Additional keyword arguments passed to the Wavelet.from_data constructor.

    Returns
    -------
    Wavelet
        Wavelet domain data

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

    mult = min(mult, Nt // 2)  # make sure K isn't bigger than ND
    phi = cp.array(phi_vec(Nf, d=nx, q=mult))
    wave = transform_wavelet_time_helper(
        cp.array(timeseries.data), Nf=Nf, Nt=Nt, phi=phi, mult=mult
    )
    return Wavelet(wave * cp.sqrt(2), time=t_bins, freq=f_bins)


def from_freq_to_wavelet(
    freqseries: FrequencySeries,
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
    nx: float = 4.0,
) -> Wavelet:
    """Transforms frequency-domain data to wavelet-domain data.

    Parameters
    ----------
    freqseries : FrequencySeries
        Frequency domain data
    Nf : int
        Number of frequency bins
    Nt : int
        Number of time bins
    nx : float, optional
        Number of standard deviations for the phi_vec, by default 4.
    **kwargs:
        Additional keyword arguments passed to the Wavelet.from_data constructor.

    Returns
    -------
    Wavelet
        Wavelet domain data

    """
    Nf, Nt = _preprocess_bins(freqseries, Nf, Nt)
    t_bins, f_bins = _get_bins(freqseries, Nf, Nt)
    phif = cp.array(phitilde_vec_norm(Nf, Nt, d=nx), dtype=backend.float_dtype)
    data = cp.array(freqseries.data, dtype=backend.complex_dtype)
    wave = transform_wavelet_freq_helper(
        data,
        Nf=Nf,
        Nt=Nt,
        phif=phif,
        float_dtype=backend.float_dtype,
        complex_dtype=backend.complex_dtype,
    )
    factor = (2 / Nf) * cp.sqrt(2)
    return Wavelet(factor * wave, time=t_bins, freq=f_bins)
