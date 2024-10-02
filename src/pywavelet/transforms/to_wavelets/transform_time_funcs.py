import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import rfft
from functools import partial

@partial(jit, static_argnames=('Nf', 'Nt', 'mult'))
def transform_wavelet_time_helper(
    data: jnp.ndarray, phi: jnp.ndarray, Nf: int, Nt: int,  mult: int
) -> jnp.ndarray:
    """Helper function to do the wavelet transform in the time domain using JAX"""
    ND = Nf * Nt
    K = mult * 2 * Nf
    
    # Pad the data
    data_pad = jnp.concatenate((data, data[:K]))
    
    # Create time bin indices
    time_bins = jnp.arange(Nt)
    
    # Vectorized __fill_wave_1
    jj_base = (time_bins[:, None] * Nf - K // 2) % ND
    jj = (jj_base + jnp.arange(K)[None, :]) % ND
    wdata = data_pad[jj] * phi[None, :]
    
    # Vectorized FFT
    wdata_trans = rfft(wdata, axis=1)
    
    # Vectorized __fill_wave_2
    wave = jnp.zeros((Nt, Nf))
    
    # Handle m=0 case
    even_mask = (time_bins % 2 == 0) & (time_bins < Nt - 1)
    wave = wave.at[time_bins[even_mask], 0].set(jnp.real(wdata_trans[even_mask, 0]) / jnp.sqrt(2))
    wave = wave.at[time_bins[even_mask] + 1, 0].set(jnp.real(wdata_trans[even_mask, Nf * mult]) / jnp.sqrt(2))
    
    # Handle other cases
    j_range = jnp.arange(1, Nf)
    t_plus_j_odd = ((time_bins[:, None] + j_range[None, :]) % 2 == 1)
    
    wave = wave.at[:, 1:].set(
        jnp.where(t_plus_j_odd,
                  -jnp.imag(wdata_trans[:, j_range * mult]),
                  jnp.real(wdata_trans[:, j_range * mult]))
    )
    
    return wave
