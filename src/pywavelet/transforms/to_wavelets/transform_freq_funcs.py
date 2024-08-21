"""helper functions for transform_freq"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.numpy.fft import fft, ifft


def transform_wavelet_freq_helper(
    data: np.ndarray, Nf: int, Nt: int, phif: jnp.ndarray
) -> np.ndarray:
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    wave = jnp.zeros((Nt, Nf))  # wavelet wavepacket transform of the signal

    DX = jnp.zeros(Nt, dtype=jnp.complex128)
    freq_strain = data.copy()  # Convert

    for f_bin in range(0, Nf + 1):
        DX = fill_wave_1(f_bin, Nt, Nf, DX, freq_strain, phif)
        DX_trans = ifft(DX, Nt)
        wave = fill_wave_2(f_bin, DX_trans, wave, Nt, Nf)

    return np.array(wave.tolist())


@jit
def fill_wave_1(
    f_bin: int,
    Nt: int,
    Nf: int,
    DX: jnp.ndarray,
    data: jnp.ndarray,
    phif: jnp.ndarray,
) -> jnp.ndarray:
    """helper for assigning DX in the main loop"""
    i_base = Nt // 2
    jj_base = f_bin * Nt // 2

    def set_initial_value(DX):
        value = jnp.where(
            (f_bin == 0) | (f_bin == Nf),
            phif[0] * data[f_bin * Nt // 2] / 2.0,
            phif[0] * data[f_bin * Nt // 2],
        )
        return DX.at[Nt // 2].set(value)

    DX = set_initial_value(DX)

    def body_fun(jj, DX):
        j = jnp.abs(jj - jj_base)
        i = i_base - jj_base + jj
        cond1 = (f_bin == Nf) & (jj > jj_base)
        cond2 = (f_bin == 0) & (jj < jj_base)
        cond3 = j == 0
        val = jnp.where(cond1 | cond2, 0.0, phif[j] * data[jj])
        DX = DX.at[i].set(jnp.where(cond3, DX[i], val))
        return DX

    return jax.lax.fori_loop(
        jj_base + 1 - Nt // 2, jj_base + Nt // 2, body_fun, DX
    )


@jit
def fill_wave_2(
    f_bin: int, DX_trans: jnp.ndarray, wave: jnp.ndarray, Nt: int, Nf: int
) -> jnp.ndarray:
    def case_0(wave):
        return wave.at[::2, 0].set(jnp.real(DX_trans[::2] * jnp.sqrt(2)))

    def case_Nf(wave):
        return wave.at[1::2, 0].set(jnp.real(DX_trans[::2] * jnp.sqrt(2)))

    def case_other(wave):
        n_range = jnp.arange(wave.shape[0])
        cond1 = (n_range + f_bin) % 2 == 1
        cond2 = f_bin % 2 == 1

        real_part = jnp.where(cond2, -jnp.imag(DX_trans), jnp.real(DX_trans))
        imag_part = jnp.where(cond2, jnp.real(DX_trans), jnp.imag(DX_trans))

        return wave.at[:, f_bin].set(jnp.where(cond1, imag_part, real_part))

    return jax.lax.cond(
        f_bin == 0,
        case_0,
        lambda w: jax.lax.cond(f_bin == Nf, case_Nf, case_other, w),
        wave,
    )
