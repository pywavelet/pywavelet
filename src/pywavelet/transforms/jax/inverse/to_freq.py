from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import fft


@partial(jit, static_argnames=("Nf", "Nt"))
def inverse_wavelet_freq_helper(
    wave_in: jnp.ndarray, phif: jnp.ndarray, Nf: int, Nt: int
) -> jnp.ndarray:
    """JAX vectorized function for inverse_wavelet_freq with corrected shapes and ranges."""
    wave_in = wave_in.T
    ND = Nf * Nt

    m_range = jnp.arange(Nf + 1)
    prefactor2s = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    n_range = jnp.arange(Nt)

    # Handle m=0 and m=Nf cases
    prefactor2s = prefactor2s.at[0].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0]
    )
    prefactor2s = prefactor2s.at[Nf].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0]
    )

    # Handle middle m cases
    m_mid = m_range[1:Nf]
    n_grid, m_grid = jnp.meshgrid(n_range, m_mid, indexing="ij")
    val = wave_in[n_grid, m_grid]
    mult2 = jnp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s = prefactor2s.at[1:Nf].set((mult2 * val).T)

    fft_prefactor2s = fft(prefactor2s, axis=1)

    res = jnp.zeros(ND // 2 + 1, dtype=jnp.complex128)

    # Unpack for m=0 and m=Nf
    i_ind_range = jnp.arange(Nt // 2)
    i_0 = i_ind_range
    ind3_0 = (2 * i_0) % Nt
    res = res.at[i_0].add(fft_prefactor2s[0, ind3_0] * phif[i_ind_range])

    i_Nf = jnp.abs(Nf * (Nt // 2) - i_ind_range)
    ind3_Nf = (2 * i_Nf) % Nt
    res = res.at[i_Nf].add(fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range])

    special_index = jnp.abs(Nf * (Nt // 2) - (Nt // 2))
    res = res.at[special_index].add(fft_prefactor2s[Nf, 0] * phif[Nt // 2])

    # Unpack for middle m values
    m_mid = m_range[1:Nf]
    i_ind_range_mid = jnp.arange(Nt // 2)
    m_grid_mid, i_ind_grid_mid = jnp.meshgrid(
        m_mid, i_ind_range_mid, indexing="ij"
    )
    i1 = (Nt // 2) * m_grid_mid - i_ind_grid_mid
    i2 = (Nt // 2) * m_grid_mid + i_ind_grid_mid
    ind31 = i1 % Nt
    ind32 = i2 % Nt

    res = res.at[i1].add(
        fft_prefactor2s[m_grid_mid, ind31] * phif[i_ind_grid_mid]
    )
    res = res.at[i2].add(
        fft_prefactor2s[m_grid_mid, ind32] * phif[i_ind_grid_mid]
    )

    # Correct the center points for middle m's
    center_indices = (Nt // 2) * m_mid
    fft_indices = center_indices % Nt
    values = fft_prefactor2s[m_mid, fft_indices] * phif[0]
    res = res.at[center_indices].set(values)

    return res
