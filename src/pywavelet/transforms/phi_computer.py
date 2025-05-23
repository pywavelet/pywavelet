"""
This module contains functions to compute the Fourier transform of the
wavelet function and its normalization. The wavelet function is defined
in the frequency domain and is used to transform time-domain data into
the wavelet domain.

Everything in this module is retured as a npfloat64 array.
"""

from functools import lru_cache

import numpy as np
from jaxtyping import Float64
from numpy.fft import ifft
from scipy.special import betainc

__all__ = ["phitilde_vec_norm", "phi_vec", "omega"]


@lru_cache(maxsize=None)
def omega(Nf: int, Nt: int) -> Float64[np.ndarray, "{Nt}//2+1"]:
    """Get the angular frequencies of the time domain signal."""
    df = 2 * np.pi / (Nf * Nt)
    return df * np.arange(0, Nt // 2 + 1, dtype=np.float64)


@lru_cache(maxsize=None)
def phitilde_vec_norm(
    Nf: int, Nt: int, d: float
) -> Float64[np.ndarray, "{Nt}//2+1"]:
    """Normalize phitilde for inverse frequency domain transform."""
    omegas = omega(Nf, Nt)
    _phi_t = _phitilde_vec(omegas, Nf, d) * np.sqrt(np.pi)
    return np.array(_phi_t)


@lru_cache(maxsize=None)
def phi_vec(
    Nf: int, d: float = 4.0, q: int = 16
) -> Float64[np.ndarray, "2*{q}*{Nf}"]:
    """get time domain phi as fourier transform of _phitilde_vec
    q: number of Nf bins over which the window extends?

    """
    insDOM = 1.0 / np.sqrt(np.pi / Nf)
    K = q * 2 * Nf
    half_K = q * Nf  # xp.int64(K/2)

    dom = 2 * np.pi / K  # max frequency is K/2*dom = pi/dt = OM
    DX = np.zeros(K, dtype=np.complex128)

    # zero frequency
    DX[0] = insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1 : half_K + 1] = _phitilde_vec(dom * np.arange(1, half_K + 1), Nf, d)
    # negative frequencies
    DX[half_K + 1 :] = _phitilde_vec(
        -dom * np.arange(half_K - 1, 0, -1), Nf, d
    )
    DX = K * ifft(DX, K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(2.0) / np.sqrt(K / dom)  # *xp.linalg.norm(phi)

    phi *= nrm
    return np.array(phi)


def _phitilde_vec(
    omega: Float64[np.ndarray, "dim"], Nf: int, d: float = 4.0
) -> Float64[np.ndarray, "dim"]:
    """Compute phi_tilde(omega_i) array, nx is filter steepness, defaults to 4.

    Eq 11 of https://arxiv.org/pdf/2009.00043.pdf (Cornish et al. 2020)

    phi(omega_i) =
        1/sqrt(2π∆F) if |omega_i| < A
        1/sqrt(2π∆F) cos(nu_d π/2 * |omega|-A / B) if A < |omega_i| < A + B

    Where nu_d = normalized incomplete beta function

    Parameters
    ----------
    ω : xp.ndarray
        Array of angular frequencies
    Nf : int
        Number of frequency bins
    d : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.

    Returns
    -------
    xp.ndarray
        Array of phi_tilde(omega_i) values

    """
    dF = 1.0 / (2 * Nf)  # NOTE: missing 1/dt?
    dOmega = 2 * np.pi * dF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_dOmega = 1.0 / np.sqrt(dOmega)

    A = dOmega / 4
    B = dOmega - 2 * A  # Cannot have B \leq 0.
    if B <= 0:
        raise ValueError("B must be greater than 0")

    phi = np.zeros(omega.size, dtype=np.float64)
    mask = (A <= np.abs(omega)) & (np.abs(omega) < A + B)  # Minor changes
    vd = (np.pi / 2.0) * _nu_d(omega[mask], A, B, d=d)  # different from paper
    phi[mask] = inverse_sqrt_dOmega * np.cos(vd)
    phi[np.abs(omega) < A] = inverse_sqrt_dOmega
    return phi


def _nu_d(
    omega: Float64[np.ndarray, "dim"], A: float, B: float, d: float = 4.0
) -> Float64[np.ndarray, "dim"]:
    """Compute the normalized incomplete beta function.

    Parameters
    ----------
    omega : np.ndarray
        Array of angular frequencies
    A : float
        Lower bound for the beta function
    B : float
        Upper bound for the beta function
    d : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.

    Returns
    -------
    np.ndarray
        Array of ν_d values

    scipy.special.betainc
    https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.special.betainc.html

    """
    x = (np.abs(omega) - A) / B
    return betainc(d, d, x)
