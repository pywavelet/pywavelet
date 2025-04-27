from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import ifft

X64_PRECISION = jax.config.jax_enable_x64

CMPLX_DTYPE = jnp.complex128 if X64_PRECISION else jnp.complex64


import logging

logger = logging.getLogger("pywavelet")


@partial(jit, static_argnames=("Nf", "Nt", "float_dtype", "complex_dtype"))
def transform_wavelet_freq_helper(
    data: jnp.ndarray,
    Nf: int,
    Nt: int,
    phif: jnp.ndarray,
    float_dtype=jnp.float64,
    complex_dtype=jnp.complex128,
) -> jnp.ndarray:
    """
    Transforms input data from the frequency domain to the wavelet domain using a
    pre-computed wavelet filter (`phif`) and performs an efficient inverse FFT.

    Parameters:
    - data (jnp.ndarray): 1D array representing the input data in the frequency domain.
    - Nf (int): Number of frequency bins.
    - Nt (int): Number of time bins. (Note: Nt * Nf == len(data))
    - phif (jnp.ndarray): Pre-computed wavelet filter for frequency components.

    Returns:
    - wave (jnp.ndarray): 2D array of wavelet-transformed data with shape (Nf, Nt).
    """
    logger.debug(
        f"[JAX TRANSFORM] Input types [data:{type(data)},{data.dtype}, phif:{type(phif)},{phif.dtype}]"
    )
    half = Nt // 2
    f_bins = jnp.arange(Nf + 1)  # [0,1,...,Nf]

    # --- 1) build the full (Nf+1, Nt) DX array ---
    # center (j = 0):
    center = phif[0] * data[f_bins * half]
    center = jnp.where((f_bins == 0) | (f_bins == Nf), center / 2.0, center)
    DX = jnp.zeros((Nf + 1, Nt), complex_dtype)
    DX = DX.at[:, half].set(center)

    # off-center (j = +/-1...+/-(half−1))
    offs = jnp.arange(1 - half, half)  # length Nt−1
    jj = f_bins[:, None] * half + offs[None, :]  # shape (Nf+1, Nt−1)
    ii = half + offs  # shape (Nt−1,)
    mask = ((f_bins[:, None] == Nf) & (offs[None, :] > 0)) | (
        (f_bins[:, None] == 0) & (offs[None, :] < 0)
    )
    vals = phif[jnp.abs(offs)] * data[jj]
    vals = jnp.where(mask, 0.0, vals)
    DX = DX.at[:, ii].set(vals)

    # --- 2) ifft along time axis ---
    DXt = jnp.fft.ifft(DX, n=Nt, axis=1)

    # --- 3) unpack into wave (Nt, Nf) ---
    wave = jnp.zeros((Nt, Nf), float_dtype)
    sqrt2 = jnp.sqrt(2.0)

    # 3a) DC into col 0, even rows
    evens = jnp.arange(0, Nt, 2)
    wave = wave.at[evens, 0].set(jnp.real(DXt[0, evens]) * sqrt2)

    # 3b) Nyquist into col 0, odd rows
    odds = jnp.arange(1, Nt, 2)
    wave = wave.at[odds, 0].set(jnp.real(DXt[Nf, evens]) * sqrt2)

    # 3c) intermediate bins 1...Nf−1
    mids = jnp.arange(1, Nf)  # [1...Nf-1]
    Dt_mid = DXt[mids, :]  # shape (Nf-1, Nt)
    real_m = jnp.real(Dt_mid).T  # (Nt, Nf-1)
    imag_m = jnp.imag(Dt_mid).T  # (Nt, Nf-1)

    odd_f = (mids % 2) == 1  # shape (Nf-1,)
    n_idx = jnp.arange(Nt)[:, None]  # (Nt,1)
    odd_n_f = ((n_idx + mids[None, :]) % 2) == 1  # (Nt, Nf-1)

    mid_vals = jnp.where(
        odd_n_f,
        jnp.where(odd_f, -imag_m, imag_m),
        jnp.where(odd_f, real_m, real_m),
    )
    wave = wave.at[:, 1:Nf].set(mid_vals)

    return wave.T
