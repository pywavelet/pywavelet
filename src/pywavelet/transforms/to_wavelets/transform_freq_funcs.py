import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.numpy.fft import ifft

@partial(jit, static_argnames=('Nf', 'Nt'))
def transform_wavelet_freq_helper(
        data: jnp.ndarray, Nf: int, Nt: int, phif: jnp.ndarray
) -> jnp.ndarray:
    # Initially all wrk being done in time-rws, freq-cols
    wave = jnp.zeros((Nt, Nf))
    f_bins = jnp.arange(Nf)

    i_base = Nt // 2
    jj_base = f_bins * Nt // 2

    initial_values = jnp.where(
        (f_bins == 0) | (f_bins == Nf),
        phif[0] * data[f_bins * Nt // 2] / 2.0,
        phif[0] * data[f_bins * Nt // 2]
    )

    DX = jnp.zeros((Nf, Nt), dtype=jnp.complex64)
    DX = DX.at[:, Nt // 2].set(initial_values)

    j_range = jnp.arange(1 - Nt // 2, Nt // 2)
    j = jnp.abs(j_range)
    i = i_base + j_range

    cond1 = (f_bins[:, None] == Nf) & (j_range[None, :] > 0)
    cond2 = (f_bins[:, None] == 0) & (j_range[None, :] < 0)
    cond3 = j[None, :] == 0

    jj = jj_base[:, None] + j_range[None, :]
    val = jnp.where(cond1 | cond2, 0.0, phif[j] * data[jj])
    DX = DX.at[:, i].set(jnp.where(cond3, DX[:, i], val))

    # Vectorized ifft
    DX_trans = ifft(DX, axis=1)

    # Vectorized __fill_wave_2_jax
    n_range = jnp.arange(Nt)
    cond1 = (n_range[:, None] + f_bins[None, :]) % 2 == 1
    cond2 = jnp.expand_dims(f_bins % 2 == 1, axis=-1) # shape: (Nf, 1)

    real_part = jnp.where(cond2, -jnp.imag(DX_trans), jnp.real(DX_trans))
    imag_part = jnp.where(cond2, jnp.real(DX_trans), jnp.imag(DX_trans))

    wave = jnp.where(cond1, imag_part.T, real_part.T)

    ## Special cases for f_bin 0 and Nf
    wave = wave.at[::2, 0].set(jnp.real(DX_trans[0, ::2] * jnp.sqrt(2)))
    wave = wave.at[1::2, -1].set(jnp.real(DX_trans[-1, ::2] * jnp.sqrt(2)))

    return wave.T