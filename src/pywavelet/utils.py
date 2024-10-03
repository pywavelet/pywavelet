from typing import Union

import jax.numpy as jnp
from scipy.interpolate import interp1d

from .transforms.types import FrequencySeries, TimeSeries, Wavelet

DATA_TYPE = Union[TimeSeries, FrequencySeries, Wavelet]


def evolutionary_psd_from_stationary_psd(
    psd: jnp.ndarray,
    psd_f: jnp.ndarray,
    f_grid: jnp.ndarray,
    t_grid: jnp.ndarray,
    dt: float,
) -> Wavelet:
    """
    PSD[ti,fi] = PSD[fi] / dt
    """
    Nt = len(t_grid)
    delta_f = f_grid[1] - f_grid[0]
    nan_val = jnp.max(psd)
    psd_grid = interp1d(
        psd_f,
        psd,
        kind="nearest",
        fill_value=nan_val,
        bounds_error=False,
    )(f_grid)

    # repeat the PSD for each time bin
    psd_grid = jnp.repeat(psd_grid[None, :], Nt, axis=0) / dt
    return Wavelet(psd_grid.T, time=t_grid, freq=f_grid)


def compute_snr(h: Wavelet, PSD: Wavelet) -> float:
    """Compute the SNR of a model h[ti,fi] given freqseries d[ti,fi] and PSD[ti,fi].

    SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi]

    Parameters
    ----------
    h : jnp.ndarray
        The model in the wavelet domain (binned in [ti,fi]).
    d : jnp.ndarray
        The freqseries in the wavelet domain (binned in [ti,fi]).
    PSD : jnp.ndarray
        The PSD in the wavelet domain (binned in [ti,fi]).

    Returns
    -------
    float
        The SNR of the model h given freqseries d and PSD.

    """
    snr_sqrd = jnp.nansum((h.data * h.data) / PSD.data)
    return jnp.sqrt(snr_sqrd)
