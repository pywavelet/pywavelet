import logging

import cupy as cp
from cupyx.scipy.fft import ifft

logger = logging.getLogger("pywavelet")


def transform_wavelet_freq_helper(
    data: cp.ndarray,
    Nf: int,
    Nt: int,
    phif: cp.ndarray,
    float_dtype=cp.float64,
    complex_dtype=cp.complex128,
) -> cp.ndarray:
    """
    Transforms input data from the frequency domain to the wavelet domain using a
    pre-computed wavelet filter (`phif`) and performs an efficient inverse FFT.

    Parameters:
    - data (cp.ndarray): 1D array representing the input data in the frequency domain.
    - Nf (int): Number of frequency bins.
    - Nt (int): Number of time bins. (Note: Nt * Nf == len(data))
    - phif (cp.ndarray): Pre-computed wavelet filter for frequency components.

    Returns:
    - wave (cp.ndarray): 2D array of wavelet-transformed data with shape (Nf, Nt).
    """

    logger.debug(
        f"[CUPY TRANSFORM] Input types [data:{type(data)}, phif:{type(phif)}, precision:{data.dtype}]"
    )

    half = Nt // 2
    f_bins = cp.arange(Nf + 1)  # [0, 1, ..., Nf]

    # 1) Build (Nf+1, Nt) DX
    # — center:
    center = phif[0] * data[f_bins * half]
    center = cp.where((f_bins == 0) | (f_bins == Nf), center * 0.5, center)
    DX = cp.zeros((Nf + 1, Nt), complex_dtype)
    DX[:, half] = center

    # — off-center (j = ±1...±(half-1))
    offs = cp.arange(1 - half, half)  # length Nt-1
    jj = f_bins[:, None] * half + offs[None, :]  # (Nf+1, Nt-1)
    ii = half + offs  # (Nt-1,)
    mask = ((f_bins[:, None] == Nf) & (offs[None, :] > 0)) | (
        (f_bins[:, None] == 0) & (offs[None, :] < 0)
    )
    vals = phif[cp.abs(offs)] * data[jj]
    vals = cp.where(mask, 0.0, vals)
    DX[:, ii] = vals

    # 2) IFFT along time
    DXt = ifft(DX, axis=1)

    # 3) Unpack into wave (Nt, Nf)
    wave = cp.zeros((Nt, Nf), float_dtype)
    sqrt2 = cp.sqrt(2.0)

    # 3a) DC into col 0, even rows
    evens = cp.arange(0, Nt, 2)
    wave[evens, 0] = cp.real(DXt[0, evens]) * sqrt2

    # 3b) Nyquist into col 0, odd rows
    odds = cp.arange(1, Nt, 2)
    wave[odds, 0] = cp.real(DXt[Nf, evens]) * sqrt2

    # 3c) intermediate bins 1...Nf-1
    mids = cp.arange(1, Nf)  # [1...Nf-1]
    Dt_mid = DXt[mids, :]  # (Nf-1, Nt)
    real_m = cp.real(Dt_mid).T  # (Nt, Nf-1)
    imag_m = cp.imag(Dt_mid).T  # (Nt, Nf-1)

    odd_f = (mids % 2) == 1  # shape (Nf-1,)
    n_idx = cp.arange(Nt)[:, None]  # (Nt,1)
    odd_nf = ((n_idx + mids[None, :]) % 2) == 1

    # Select real vs imag and sign exactly as in __fill_wave_2
    mid_vals = cp.where(
        odd_nf,
        cp.where(odd_f, -imag_m, imag_m),
        cp.where(odd_f, real_m, real_m),
    )
    wave[:, 1:Nf] = mid_vals

    return wave.T
