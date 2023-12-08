"""Wavelet domain SNR

SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi] ],

where h_hat[ti,fi] is the unit normalized wavelet transform of the model:
h_hat[ti,fi] = h[ti,fi] / sqrt(<h[ti,fi] | h[ti,fi] >)

NOTE: to maximize over masses and spins we require some additional steps....


"""

import numpy as np

from pywavelet.logger import logger
from pywavelet.transforms.types import Wavelet


def compute_snr(h: Wavelet, d: Wavelet, PSD: Wavelet) -> float:
    """Compute the SNR of a model h[ti,fi] given data d[ti,fi] and PSD[ti,fi].

    SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi]

    Parameters
    ----------
    h : np.ndarray
        The model in the wavelet domain (binned in [ti,fi]).
    d : np.ndarray
        The data in the wavelet domain (binned in [ti,fi]).
    PSD : np.ndarray
        The PSD in the wavelet domain (binned in [ti,fi]).

    Returns
    -------
    float
        The SNR of the model h given data d and PSD.

    """
    # h = h.data
    # d = d.data
    # PSD = PSD.data

    h = h.data
    h_hat = h / np.sqrt(np.tensordot(h, h))
    d_hat = d.data / PSD.data

    # mask any nans/inf
    mask = (
        np.isnan(h_hat) | np.isinf(h_hat) | np.isnan(d_hat) | np.isinf(d_hat)
    )
    h_hat[mask] = 0
    d_hat[mask] = 0

    # if mask size > 2% of d_hat size, raise warning
    if np.sum(mask) > 0.02 * d_hat.size:
        logger.warning(
            f"{np.sum(mask)} / {d_hat.size} elements masked in SNR computation"
        )

    return np.tensordot(h_hat, d_hat).item()
