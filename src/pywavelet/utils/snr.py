"""Wavelet domain SNR

SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi] ],

where h_hat[ti,fi] is the unit normalized wavelet transform of the model:
h_hat[ti,fi] = h[ti,fi] / sqrt(<h[ti,fi] | h[ti,fi] >)

NOTE: to maximize over masses and spins we require some additional steps....


"""

import numpy as np


def compute_snr(h: np.ndarray, d: np.ndarray, PSD: np.ndarray) -> float:
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
    h_hat = h / np.sqrt(np.tensordot(h.T, h))
    return np.tensordot(h_hat.T, d / PSD)
