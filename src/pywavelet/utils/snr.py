"""Wavelet domain SNR

SNR(h) = Sum_{ti,fi} [ h_hat[ti,fi] d[ti,fi] / PSD[ti,fi] ],

where h_hat[ti,fi] is the unit normalized wavelet transform of the model:
h_hat[ti,fi] = h[ti,fi] / sqrt(<h[ti,fi] | h[ti,fi] >)

NOTE: to maximize over masses and spins we require some additional steps....


"""

import numpy as np

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
    snr_sqrd = np.nansum((h * h) / PSD)
    return np.sqrt(snr_sqrd)


def compute_frequency_optimal_snr(h_freq, psd, duration):
    snr_sqrd = __noise_weighted_inner_product(
        aa=h_freq, bb=h_freq, power_spectral_density=psd, duration=duration
    ).real
    return np.sqrt(snr_sqrd)


def __noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    =======
    Noise-weighted inner product.
    """
    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 / duration * np.sum(integrand)
