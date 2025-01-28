from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import rfft


@partial(jit, static_argnames=("Nf", "Nt", "mult"))
def transform_wavelet_time_helper(
    data: jnp.ndarray, phi: jnp.ndarray, Nf: int, Nt: int, mult: int
) -> jnp.ndarray:
    """Helper function to do the wavelet transform in the time domain using JAX"""
    # Define constants
    ND = Nf * Nt
    K = mult * 2 * Nf

    # Pad the data with K extra values
    data_pad = jnp.concatenate((data, data[:K]))

    # Generate time bin indices
    time_bins = jnp.arange(Nt)
    jj_base = (time_bins[:, None] * Nf - K // 2) % ND
    jj = (jj_base + jnp.arange(K)[None, :]) % ND

    # Apply the window (phi) to the data
    wdata = data_pad[jj] * phi[None, :]

    # Perform FFT on the windowed data
    wdata_trans = rfft(wdata, axis=1)

    # Initialize the wavelet transform result
    wave = jnp.zeros((Nt, Nf))

    # Handle m=0 case for even time bins
    even_mask = (time_bins % 2 == 0) & (time_bins < Nt - 1)
    even_indices = jnp.nonzero(even_mask, size=even_mask.shape[0])[0]

    # Update wave for m=0 using even time bins
    wave = wave.at[even_indices, 0].set(
        jnp.real(wdata_trans[even_indices, 0]) / jnp.sqrt(2)
    )
    wave = wave.at[even_indices + 1, 0].set(
        jnp.real(wdata_trans[even_indices, Nf * mult]) / jnp.sqrt(2)
    )

    # Handle other cases (j > 0) using vectorized operations
    j_range = jnp.arange(1, Nf)
    odd_condition = (time_bins[:, None] + j_range[None, :]) % 2 == 1

    wave = wave.at[:, 1:].set(
        jnp.where(
            odd_condition,
            -jnp.imag(wdata_trans[:, j_range * mult]),
            jnp.real(wdata_trans[:, j_range * mult]),
        )
    )

    return wave.T
