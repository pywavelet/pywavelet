import numpy as np
import scipy

from .. import fft_funcs as fft

PI = np.pi


def phitilde_vec(om: np.ndarray, Nf: int, nx=4.0) -> np.ndarray:
    """compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    # Note: Pi is Nyquist angular frequency
    DOM = PI / Nf  # 2 pi times DF
    insDOM = 1.0 / np.sqrt(DOM)
    B = PI / (2 * Nf)
    A = (DOM - B) / 2
    z = np.zeros(om.size)

    mask = (np.abs(om) >= A) & (np.abs(om) < A + B)

    x = (np.abs(om[mask]) - A) / B
    y = scipy.special.betainc(nx, nx, x)
    z[mask] = insDOM * np.cos(PI / 2.0 * y)

    z[np.abs(om) < A] = insDOM
    return z


def phitilde_vec_norm(Nf: int, Nt: int, nx: int) -> np.ndarray:
    """Normalize phitilde for inverse frequency domain transform."""

    # Calculate the frequency values
    ND = Nf * Nt
    omegas = 2 * np.pi / ND * np.arange(0, Nt // 2 + 1)

    # Calculate the unnormalized phitilde (u_phit)
    u_phit = phitilde_vec(omegas, Nf, nx)

    # Normalize the phitilde
    nrm_fctor = np.sqrt(
        (2 * np.sum(u_phit[1:] ** 2) + u_phit[0] ** 2) * 2 * PI / ND
    )
    nrm_fctor /= PI ** (3 / 2) / PI

    return u_phit / nrm_fctor


def phi_vec(Nf: int, nx: int = 4.0, mult: int = 16) -> np.ndarray:
    """get time domain phi as fourier transform of phitilde_vec"""
    insDOM = 1.0 / np.sqrt(PI / Nf)
    K = mult * 2 * Nf
    half_K = mult * Nf  # np.int64(K/2)

    dom = 2 * PI / K  # max frequency is K/2*dom = pi/dt = OM

    DX = np.zeros(K, dtype=np.complex128)

    # zero frequency
    DX[0] = insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1 : half_K + 1] = phitilde_vec(dom * np.arange(1, half_K + 1), Nf, nx)
    # negative frequencies
    DX[half_K + 1 :] = phitilde_vec(
        -dom * np.arange(half_K - 1, 0, -1), Nf, nx
    )
    DX = K * fft.ifft(DX, K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(K / dom)  # *np.linalg.norm(phi)

    fac = np.sqrt(2.0) / nrm
    phi *= fac
    return phi
