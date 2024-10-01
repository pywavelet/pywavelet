import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import fft
from functools import partial

@partial(jit, static_argnums=(2, 3, 4))
def inverse_wavelet_time_helper(wave_in: jnp.ndarray, phi: jnp.ndarray, Nf: int, Nt: int, mult: int) -> jnp.ndarray:
    """Perform inverse wavelet transform in time domain."""
    ND = Nf * Nt
    K = mult * 2 * Nf

    result = jnp.zeros(ND + K + Nf)
    even_timesteps = jnp.arange(0, Nt, 2)

    packed_coefficients = __pack_wavelet_coefficients(wave_in, Nf, Nt, even_timesteps)
    transformed_coefficients = fft(packed_coefficients, axis=1)
    result = __unpack_wavelet_coefficients(transformed_coefficients, phi, Nf, Nt, K, even_timesteps, result)

    result = apply_boundary_conditions(result, K, Nf, ND)

    return result[:ND]

@jit
def __pack_wavelet_coefficients(wave_in: jnp.ndarray, Nf: int, Nt: int, even_timesteps: jnp.ndarray) -> jnp.ndarray:
    """Pack wavelet coefficients for efficient processing."""
    packed = jnp.zeros((len(even_timesteps), 2 * Nf), dtype=jnp.complex128)

    # Handle special cases
    packed = packed.at[:, 0].set(jnp.sqrt(2) * wave_in[even_timesteps, 0])
    packed = packed.at[:, Nf].set(jnp.where(even_timesteps + 1 < Nt, jnp.sqrt(2) * wave_in[even_timesteps + 1, 0], 0))

    # Handle general cases
    freq_indices = jnp.arange(0, Nf - 2, 2)
    packed = __pack_even_odd_frequencies(packed, wave_in, even_timesteps, freq_indices, Nf)

    # Handle last frequency
    packed = packed.at[:, Nf - 1].set(1j * (wave_in[even_timesteps, Nf - 1] - wave_in[even_timesteps + 1, Nf - 1]))
    packed = packed.at[:, Nf + 1].set(-1j * (wave_in[even_timesteps, Nf - 1] + wave_in[even_timesteps + 1, Nf - 1]))

    return packed

@jit
def __pack_even_odd_frequencies(packed: jnp.ndarray, wave_in: jnp.ndarray, even_timesteps: jnp.ndarray, freq_indices: jnp.ndarray, Nf: int) -> jnp.ndarray:
    """Pack even and odd frequencies."""
    even_freq = freq_indices + 2
    odd_freq = freq_indices + 1

    packed = packed.at[:, even_freq].set(wave_in[even_timesteps[:, None], even_freq] - wave_in[even_timesteps[:, None] + 1, even_freq])
    packed = packed.at[:, 2 * Nf - even_freq].set(wave_in[even_timesteps[:, None], even_freq] + wave_in[even_timesteps[:, None] + 1, even_freq])

    packed = packed.at[:, odd_freq].set(1j * (wave_in[even_timesteps[:, None], odd_freq] - wave_in[even_timesteps[:, None] + 1, odd_freq]))
    packed = packed.at[:, 2 * Nf - odd_freq].set(-1j * (wave_in[even_timesteps[:, None], odd_freq] + wave_in[even_timesteps[:, None] + 1, odd_freq]))

    return packed

@jit
def __unpack_wavelet_coefficients(transformed_coeffs: jnp.ndarray, phi: jnp.ndarray, Nf: int, Nt: int, K: int, even_timesteps: jnp.ndarray, result: jnp.ndarray) -> jnp.ndarray:
    """Unpack wavelet coefficients after transformation."""
    ND = Nf * Nt
    real_part = jnp.concatenate([jnp.real(transformed_coeffs), jnp.real(transformed_coeffs)], axis=1)
    imag_part = jnp.concatenate([jnp.imag(jnp.roll(transformed_coeffs, Nf, axis=1))] * 2, axis=1)

    base_indices = ((-K // 2 + even_timesteps[:, None] * Nf + ND) % (2 * Nf))[:, None]
    base_positions = ((-K // 2 + even_timesteps[:, None] * Nf) % ND)[:, None]

    for offset in range(0, K, 2 * Nf):
        result = __update_result(result, real_part, imag_part, phi, base_indices, base_positions, offset, Nf, ND)

    return result

@jit
def __update_result(result: jnp.ndarray, real_part: jnp.ndarray, imag_part: jnp.ndarray, phi: jnp.ndarray,
                    base_indices: jnp.ndarray, base_positions: jnp.ndarray, offset: int, Nf: int, ND: int) -> jnp.ndarray:
    """Update result array with unpacked coefficients."""
    local_indices = jnp.arange(2 * Nf)
    indices = (base_indices + local_indices) % (4 * Nf)
    positions = (base_positions + offset + local_indices) % ND

    result = result.at[positions].add(phi[offset + local_indices] * real_part[jnp.arange(len(base_indices))[:, None], indices])
    result = result.at[(positions + Nf) % ND].add(phi[offset + local_indices] * imag_part[jnp.arange(len(base_indices))[:, None], indices])

    return result

@jit
def apply_boundary_conditions(result: jnp.ndarray, K: int, Nf: int, ND: int) -> jnp.ndarray:
    """Apply boundary conditions to the result."""
    overlap = jnp.minimum(K + Nf, ND)
    result = result.at[:overlap].add(result[ND:ND + overlap])
    
    if K + Nf > ND:
        result = result.at[:K + Nf - ND].add(result[2 * ND:ND + K * Nf])
    
    return result

