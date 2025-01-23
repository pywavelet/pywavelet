from ..backend import xp, PI, betainc, ifft



def phitilde_vec(
    omega: xp.ndarray, Nf: int, d: float = 4.0
) -> xp.ndarray:
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
    dOmega = 2 * PI * dF  # Near Eq 10 # 2 pi times DF
    inverse_sqrt_dOmega = 1.0 / xp.sqrt(dOmega)

    A = dOmega / 4
    B = dOmega - 2 * A  # Cannot have B \leq 0.
    if B <= 0:
        raise ValueError("B must be greater than 0")

    phi = xp.zeros(omega.size)
    mask = (A <= xp.abs(omega)) & (xp.abs(omega) < A + B)  # Minor changes
    vd = (PI / 2.0) * __nu_d(omega[mask], A, B, d=d)  # different from paper
    phi[mask] = inverse_sqrt_dOmega * xp.cos(vd)
    phi[xp.abs(omega) < A] = inverse_sqrt_dOmega
    return phi


def __nu_d(
    omega: xp.ndarray, A: float, B: float, d: float = 4.0
) -> xp.ndarray:
    """Compute the normalized incomplete beta function.

    Parameters
    ----------
    ω : xp.ndarray
        Array of angular frequencies
    A : float
        Lower bound for the beta function
    B : float
        Upper bound for the beta function
    d : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.

    Returns
    -------
    xp.ndarray
        Array of ν_d values

    scipy.special.betainc
    https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.special.betainc.html

    """
    x = (xp.abs(omega) - A) / B
    return betainc(d, d, x) / betainc(d, d, 1)


def phitilde_vec_norm(Nf: int, Nt: int,  d: float) -> xp.ndarray:
    """Normalize phitilde for inverse frequency domain transform."""

    # Calculate the frequency values
    ND = Nf * Nt
    omegas = 2 * xp.pi / ND * xp.arange(0, Nt // 2 + 1)

    # Calculate the unnormalized phitilde (u_phit)
    u_phit = phitilde_vec(omegas, Nf,  d)

    # Normalize the phitilde
    normalising_factor = PI ** (-1 / 2)  # Ollie's normalising factor

    # Notes: this is the overall normalising factor that is different from Cornish's paper
    # It is the only way I can force this code to be consistent with our work in the
    # frequency domain. First note that

    # old normalising factor -- This factor is absolutely ridiculous. Why!?
    # Matt_normalising_factor = np.sqrt(
    #     (2 * np.sum(u_phit[1:] ** 2) + u_phit[0] ** 2) * 2 * PI / ND
    # )
    # Matt_normalising_factor /= PI**(3/2)/PI

    # The expression above is equal to np.pi**(-1/2) after working through the maths.
    # I have pulled (2/Nf) from __init__.py (from freq to wavelet) into the normalsiing
    # factor here. I thnk it's cleaner to have ONE normalising constant. Avoids confusion
    # and it is much easier to track.

    # TODO: understand the following:
    # (2 * np.sum(u_phit[1:] ** 2) + u_phit[0] ** 2) = 0.5 * Nt / dOmega
    # Matt_normalising_factor is equal to 1/sqrt(pi)... why is this computed?
    # in such a stupid way?

    return u_phit / (normalising_factor)


def phi_vec(Nf: int, d: float = 4.0, q: int = 16) -> xp.ndarray:
    """get time domain phi as fourier transform of phitilde_vec"""
    insDOM = 1.0 / xp.sqrt(PI / Nf)
    K = q * 2 * Nf
    half_K = q * Nf  # xp.int64(K/2)

    dom = 2 * PI / K  # max frequency is K/2*dom = pi/dt = OM

    DX = xp.zeros(K, dtype=xp.complex128)

    # zero frequency
    DX[0] = insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1 : half_K + 1] = phitilde_vec(
        dom * xp.arange(1, half_K + 1), Nf, d
    )
    # negative frequencies
    DX[half_K + 1 :] = phitilde_vec(
        -dom * xp.arange(half_K - 1, 0, -1), Nf,  d
    )
    DX = K * ifft(DX, K)

    phi = xp.zeros(K)
    phi[0:half_K] = xp.real(DX[half_K:K])
    phi[half_K:] = xp.real(DX[0:half_K])

    nrm = xp.sqrt(K / dom)  # *xp.linalg.norm(phi)

    fac = xp.sqrt(2.0) / nrm
    phi *= fac
    return phi
