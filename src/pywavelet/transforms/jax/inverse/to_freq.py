import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import fft

from functools import partial


@partial(jit, static_argnames=('Nf', 'Nt'))
def inverse_wavelet_freq_helper(
        wave_in: jnp.ndarray, phif: jnp.ndarray, Nf: int, Nt: int
) -> jnp.ndarray:
    """JAX vectorized function for inverse_wavelet_freq"""
    wave_in = wave_in.T
    ND = Nf * Nt

    m_range = jnp.arange(Nf + 1)
    prefactor2s = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)

    n_range = jnp.arange(Nt)

    # m == 0 case
    prefactor2s = prefactor2s.at[0].set(2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0])

    # m == Nf case
    prefactor2s = prefactor2s.at[Nf].set(2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0])

    # Other m cases
    m_mid = m_range[1:Nf]
    n_grid, m_grid = jnp.meshgrid(n_range, m_mid)
    val = wave_in[n_grid, m_grid]
    mult2 = jnp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s = prefactor2s.at[1:Nf].set(mult2 * val)

    # Vectorized FFT
    fft_prefactor2s = fft(prefactor2s, axis=1)

    # Vectorized __unpack_wave_inverse
    ## TODO: Check with Giorgio
    # ND or ND // 2 + 1?
    # https://github.com/pywavelet/pywavelet/blob/63151a47cde9edc14f1e7e0bf17f554e78ad257c/src/pywavelet/transforms/from_wavelets/inverse_wavelet_freq_funcs.py
    res = jnp.zeros(ND, dtype=jnp.complex128)

    # m == 0 or m == Nf cases
    i_ind_range = jnp.arange(Nt // 2)
    i_0 = jnp.abs(i_ind_range)
    i_Nf = jnp.abs(Nf * Nt // 2 - i_ind_range)
    ind3_0 = (2 * i_0) % Nt
    ind3_Nf = (2 * i_Nf) % Nt

    res = res.at[i_0].add(fft_prefactor2s[0, ind3_0] * phif[i_ind_range])
    res = res.at[i_Nf].add(fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range])

    # Special case for m == Nf
    res = res.at[Nf * Nt // 2].add(fft_prefactor2s[Nf, 0] * phif[Nt // 2])

    # Other m cases
    m_mid = m_range[1:Nf]
    i_ind_range = jnp.arange(Nt // 2 + 1)
    m_grid, i_ind_grid = jnp.meshgrid(m_mid, i_ind_range)

    i1 = Nt // 2 * m_grid - i_ind_grid
    i2 = Nt // 2 * m_grid + i_ind_grid
    ind31 = (Nt // 2 * m_grid - i_ind_grid) % Nt
    ind32 = (Nt // 2 * m_grid + i_ind_grid) % Nt

    res = res.at[i1].add(fft_prefactor2s[m_grid, ind31] * phif[i_ind_grid])
    res = res.at[i2].add(fft_prefactor2s[m_grid, ind32] * phif[i_ind_grid])

    return res