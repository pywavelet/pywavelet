import numpy as np
import scipy

from .. import fft_funcs as fft

PI = np.pi


def phitilde_vec(ω: np.ndarray, Nf: int, dt: float, d=4.0) -> np.ndarray:
    """Compute phi_tilde(omega_i) array, nx is filter steepness, defaults to 4.

    Eq 11 of https://arxiv.org/pdf/2009.00043.pdf (Cornish et al. 2020)

    phi(omega_i) =
        1/sqrt(2π∆F) if |omega_i| < A
        1/sqrt(2π∆F) cos(nu_d π/2 * |omega|-A / B) if A < |omega_i| < A + B

    Where nu_d = normalized incomplete beta function



    Parameters
    ----------
    ω : np.ndarray
        Array of angular frequencies
    Nf : int
        Number of frequency bins
    d : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.

    Returns
    -------
    np.ndarray
        Array of phi_tilde(omega_i) values

    """
    ΔF = 1.0 / (2 * Nf)  # NOTE: added missing 1/dt, EQ 7 in Cornish paper?
    ΔΩ = 2 * PI * ΔF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_ΔΩ = 1.0 / np.sqrt(ΔΩ)

    B = ΔΩ / 2
    A = ΔΩ / 4

    phi = np.zeros(ω.size)
    mask = (A <= np.abs(ω)) & (np.abs(ω) < A + B)
    vd = (PI / 2.0) * __νd(ω[mask], A, B, d=d)  # different from paper
    phi[mask] = inverse_sqrt_ΔΩ * np.cos(vd)
    phi[np.abs(ω) < A] = inverse_sqrt_ΔΩ
    return phi


def __νd(ω, A, B, d=4.0):
    """Compute the normalized incomplete beta function.

    Parameters
    ----------
    ω : np.ndarray
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
    x = (np.abs(ω) - A) / B
    numerator = scipy.special.betainc(d, d, x)
    denominator = scipy.special.betainc(d, d, 1)
    return numerator / denominator


def phitilde_vec_norm(Nf: int, Nt: int, dt: float, d: int) -> np.ndarray:
    """Normalize phitilde for inverse frequency domain transform."""

    # Calculate the frequency values
    ND = Nf * Nt
    omegas = 2 * np.pi / ND * np.arange(0, Nt // 2 + 1)

    # Calculate the unnormalized phitilde (u_phit)
    u_phit = phitilde_vec(omegas, Nf, dt, d)

    # Normalize the phitilde
    nrm_fctor = np.sqrt(
        (2 * np.sum(u_phit[1:] ** 2) + u_phit[0] ** 2) * 2 * PI / ND
    )
    nrm_fctor /= PI ** (3 / 2) / PI

    return u_phit / (nrm_fctor)


def phi_vec(Nf: int, dt, d: float = 4.0, q: int = 16) -> np.ndarray:
    """get time domain phi as fourier transform of phitilde_vec"""
    insDOM = 1.0 / np.sqrt(PI / Nf)
    K = q * 2 * Nf
    half_K = q * Nf  # np.int64(K/2)

    dom = 2 * PI / K  # max frequency is K/2*dom = pi/dt = OM

    DX = np.zeros(K, dtype=np.complex128)

    # zero frequency
    DX[0] = insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1 : half_K + 1] = phitilde_vec(
        dom * np.arange(1, half_K + 1), Nf, dt, d
    )
    # negative frequencies
    DX[half_K + 1 :] = phitilde_vec(
        -dom * np.arange(half_K - 1, 0, -1), Nf, dt, d
    )
    DX = K * fft.ifft(DX, K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(K / dom)  # *np.linalg.norm(phi)

    fac = np.sqrt(2.0) / nrm
    phi *= fac
    return phi
