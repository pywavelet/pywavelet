"""Wavelet domain SNR

SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi] ],

where h_hat[ti,fi] is the unit normalized wavelet transform of the model:
h_hat[ti,fi] = h[ti,fi] / sqrt(<h[ti,fi] | h[ti,fi] >)

NOTE: to maximize over masses and spins we require some additional steps....


"""

import numpy as np

from pywavelet.logger import logger
from pywavelet.transforms.types import Wavelet


def compute_snr(h: Wavelet, PSD: Wavelet) -> float:
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
    snr_sqrd = np.nansum((h.data * h.data) / PSD.data**2)
    snr_sqrd = snr_sqrd / (h.Nf * h.Nt * np.pi)
    return np.sqrt(snr_sqrd)
