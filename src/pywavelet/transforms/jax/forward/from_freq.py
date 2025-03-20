from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import ifft

import logging

logger = logging.getLogger('pywavelet')


@partial(jit, static_argnames=("Nf", "Nt"))
def transform_wavelet_freq_helper(
    data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray
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

    logger.debug(f"[JAX TRANSFORM] Input types [data:{type(data)}, phif:{type(phif)}]")

    # Initialize the wavelet output array with zeros (time-rows, frequency-columns)
    wave = jnp.zeros((Nt, Nf))
    f_bins = jnp.arange(Nf)  # Frequency bin indices

    # Compute base indices for time (i_base) and frequency (jj_base)
    i_base = Nt // 2
    jj_base = f_bins * Nt // 2

    # Set initial values for the center of the transformation
    initial_values = jnp.where(
        (f_bins == 0)
        | (f_bins == Nf),  # Edge cases: DC (f=0) and Nyquist (f=Nf)
        phif[0] * data[f_bins * Nt // 2] / 2.0,  # Adjust for symmetry
        phif[0] * data[f_bins * Nt // 2],
    )

    # Initialize a 2D array to store intermediate FFT input values
    DX = jnp.zeros(
        (Nf, Nt), dtype=jnp.complex64
    )  # TODO: Check dtype -- is complex64 sufficient?
    DX = DX.at[:, Nt // 2].set(
        initial_values
    )  # Set initial values at the center of the transformation (2 sided FFT)

    # Compute time indices for all offsets around the midpoint
    j_range = jnp.arange(
        1 - Nt // 2, Nt // 2
    )  # Time offsets (centered around zero)
    j = jnp.abs(j_range)  # Absolute offset indices
    i = i_base + j_range  # Time indices in output array

    # Compute conditions for edge cases
    cond1 = (f_bins[:, None] == Nf) & (j_range[None, :] > 0)  # Nyquist
    cond2 = (f_bins[:, None] == 0) & (j_range[None, :] < 0)  # DC
    cond3 = j[None, :] == 0  # Center of the transformation (no offset)

    # Compute frequency indices for the input data
    jj = jj_base[:, None] + j_range[None, :]  # Frequency offsets
    val = jnp.where(
        cond1 | cond2, 0.0, phif[j] * data[jj]
    )  # Wavelet filter application
    DX = DX.at[:, i].set(
        jnp.where(cond3, DX[:, i], val)
    )  # Update DX with computed values
    # At this point, DX contains the data FFT'd with the wavelet filter
    # (each row is a frequency bin, each column is a time bin)

    # Perform the inverse FFT along the time dimension
    DX_trans = ifft(DX, axis=1)

    # Fill the wavelet output array based on the inverse FFT results
    n_range = jnp.arange(Nt)  # Time indices
    cond1 = (
        n_range[:, None] + f_bins[None, :]
    ) % 2 == 1  # Odd/even alternation
    cond2 = jnp.expand_dims(f_bins % 2 == 1, axis=-1)  # Odd frequency bins

    # Assign real and imaginary parts based on conditions
    real_part = jnp.where(cond2, -jnp.imag(DX_trans), jnp.real(DX_trans))
    imag_part = jnp.where(cond2, jnp.real(DX_trans), jnp.imag(DX_trans))
    wave = jnp.where(cond1, imag_part.T, real_part.T)

    # Special cases for frequency bins 0 (DC) and Nf (Nyquist)
    wave = wave.at[::2, 0].set(
        jnp.real(DX_trans[0, ::2] * jnp.sqrt(2))
    )  # DC component
    wave = wave.at[1::2, -1].set(
        jnp.real(DX_trans[-1, ::2] * jnp.sqrt(2))
    )  # Nyquist component

    # Return the wavelet-transformed array (transposed for freq-major layout)
    return wave.T  # (Nt, Nf) -> (Nf, Nt)
