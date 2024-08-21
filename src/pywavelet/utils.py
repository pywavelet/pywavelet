from typing import Union

import numpy as np
from scipy.interpolate import interp1d

from .transforms.types import FrequencySeries, TimeSeries, Wavelet

DATA_TYPE = Union[TimeSeries, FrequencySeries, Wavelet]


def evolutionary_psd_from_stationary_psd(
    psd: np.ndarray,
    psd_f: np.ndarray,
    f_grid: np.ndarray,
    t_grid: np.ndarray,
    dt: float,
) -> Wavelet:
    """
    PSD[ti,fi] = PSD[fi] / dt
    """
    Nt = len(t_grid)
    delta_f = f_grid[1] - f_grid[0]
    nan_val = np.max(psd)
    psd_grid = interp1d(
        psd_f,
        psd,
        kind="nearest",
        fill_value=nan_val,
        bounds_error=False,
    )(f_grid)

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0) / dt
    w = Wavelet.from_data(psd_grid, time_grid=t_grid, freq_grid=f_grid)
    return w


def compute_snr(h: Wavelet, PSD: Wavelet) -> float:
    """Compute the SNR of a model h[ti,fi] given freqseries d[ti,fi] and PSD[ti,fi].

    SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi]

    Parameters
    ----------
    h : np.ndarray
        The model in the wavelet domain (binned in [ti,fi]).
    d : np.ndarray
        The freqseries in the wavelet domain (binned in [ti,fi]).
    PSD : np.ndarray
        The PSD in the wavelet domain (binned in [ti,fi]).

    Returns
    -------
    float
        The SNR of the model h given freqseries d and PSD.

    """
    snr_sqrd = np.nansum((h * h) / PSD)
    return np.sqrt(snr_sqrd)
