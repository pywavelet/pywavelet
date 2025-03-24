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
    # Transpose to match the NumPy version.
    wave_in = wave_in.T
    ND = Nf * Nt

    # Allocate prefactor2s for each m (shape: (Nf+1, Nt)).
    m_range = jnp.arange(Nf + 1)
    prefactor2s = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    n_range = jnp.arange(Nt)

    # m == 0 case
    prefactor2s = prefactor2s.at[0].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0]
    )

    # m == Nf case
    prefactor2s = prefactor2s.at[Nf].set(
        2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0]
    )

    # Other m cases: use meshgrid for vectorization.
    m_mid = m_range[1:Nf]
    # Create grids: n_grid (columns) and m_grid (rows)
    n_grid, m_grid = jnp.meshgrid(n_range, m_mid)
    # Index the transposed wave_in using (n, m) as in the NumPy version.
    val = wave_in[n_grid, m_grid]
    # Apply the alternating multiplier based on (n+m) parity.
    mult2 = jnp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s = prefactor2s.at[1:Nf].set(mult2 * val)

    # Apply FFT along axis 1 for all m.
    fft_prefactor2s = fft(prefactor2s, axis=1)

    # Allocate the result array with corrected shape.
    res = jnp.zeros(ND // 2 + 1, dtype=jnp.complex128)

    # Unpacking for m == 0 and m == Nf cases:
    i_ind_range = jnp.arange(Nt // 2)
    i_0 = jnp.abs(i_ind_range)  # for m == 0: i = i_ind_range
    i_Nf = jnp.abs(Nf * (Nt // 2) - i_ind_range)
    ind3_0 = (2 * i_0) % Nt
    ind3_Nf = (2 * i_Nf) % Nt

    res = res.at[i_0].add(fft_prefactor2s[0, ind3_0] * phif[i_ind_range])
    res = res.at[i_Nf].add(fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range])
    # Special case for m == Nf (ensure the Nyquist frequency is updated correctly)
    special_index = jnp.abs(Nf * (Nt // 2) - (Nt // 2))
    res = res.at[special_index].add(fft_prefactor2s[Nf, 0] * phif[Nt // 2])

    # Unpacking for m in (1, ..., Nf-1)
    m_mid = m_range[1:Nf]
    # Use range [0, Nt//2) to match the loop in NumPy version.
    i_ind_range_mid = jnp.arange(Nt // 2)
    # Create meshgrid for vectorized computation.
    m_grid_mid, i_ind_grid_mid = jnp.meshgrid(
        m_mid, i_ind_range_mid, indexing="ij"
    )

    # Compute indices i1 and i2 following the NumPy logic.
    i1 = (Nt // 2) * m_grid_mid - i_ind_grid_mid
    i2 = (Nt // 2) * m_grid_mid + i_ind_grid_mid
    # Compute the wrapped indices for FFT results.
    ind31 = i1 % Nt
    ind32 = i2 % Nt

    # Update result array using vectorized adds.
    # Note: You might need to adjust this further if your target res shape is non-trivial,
    # because here we assume that i1 and i2 indices fall within the allocated result shape.
    res = res.at[i1].add(
        fft_prefactor2s[m_grid_mid, ind31] * phif[i_ind_grid_mid]
    )
    res = res.at[i2].add(
        fft_prefactor2s[m_grid_mid, ind32] * phif[i_ind_grid_mid]
    )

    return res
