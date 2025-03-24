"""functions for computing the inverse wavelet transforms"""

import numpy as np
from numba import njit
from numpy import fft


def inverse_wavelet_freq_helper_fast(
    wave_in: np.ndarray, phif: np.ndarray, Nf: int, Nt: int
) -> np.ndarray:
    """jit compatible loop for inverse_wavelet_freq"""
    wave_in = wave_in.T
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt, np.complex128)
    res = np.zeros(ND // 2 + 1, dtype=np.complex128)
    __core(Nf, Nt, prefactor2s, wave_in, phif, res)

    return res


def __core(
    Nf: int,
    Nt: int,
    prefactor2s: np.ndarray,
    wave_in: np.ndarray,
    phif: np.ndarray,
    res: np.ndarray,
) -> None:
    for m in range(0, Nf + 1):
        __pack_wave_inverse(m, Nt, Nf, prefactor2s, wave_in)
        fft_prefactor2s = np.fft.fft(prefactor2s)
        __unpack_wave_inverse(m, Nt, Nf, phif, fft_prefactor2s, res)


@njit()
def __pack_wave_inverse(
    m: int, Nt: int, Nf: int, prefactor2s: np.ndarray, wave_in: np.ndarray
) -> None:
    """helper for fast frequency domain inverse transform to prepare for fourier transform"""
    if m == 0:
        for n in range(0, Nt):
            prefactor2s[n] = 2 ** (-1 / 2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(0, Nt):
            prefactor2s[n] = 2 ** (-1 / 2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(0, Nt):
            val = wave_in[n, m]
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2 * val


@njit()
def __unpack_wave_inverse(
    m: int,
    Nt: int,
    Nf: int,
    phif: np.ndarray,
    fft_prefactor2s: np.ndarray,
    res: np.ndarray,
) -> None:
    """helper for unpacking results of frequency domain inverse transform"""

    if m == 0 or m == Nf:
        for i_ind in range(0, Nt // 2):
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = (2 * i) % Nt
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == Nf:
            i_ind = Nt // 2
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (Nt // 2 * m) % Nt
        ind32 = (Nt // 2 * m) % Nt
        for i_ind in range(0, Nt // 2):
            i1 = Nt // 2 * m - i_ind
            i2 = Nt // 2 * m + i_ind
            # assert ind31 == i1%Nt
            # assert ind32 == i2%Nt
            res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31 < 0:
                ind31 = Nt - 1
            if ind32 == Nt:
                ind32 = 0
        res[Nt // 2 * m] = fft_prefactor2s[(Nt // 2 * m) % Nt] * phif[0]


#
# # @njit
# def inverse_wavelet_freq_helper_fast_version2(
#         wave_in: np.ndarray, phif: np.ndarray, Nf: int, Nt: int
# ) -> np.ndarray:
#     wave_in = wave_in.T
#     ND = Nf * Nt
#     prefactor2s = np.zeros((Nf + 1, Nt), dtype=np.complex128)
#     n_range = np.arange(Nt)
#
#     # m == 0 case
#     indices = (2 * n_range) % Nt
#     prefactor2s[0] = (2 ** (-0.5)) * wave_in[indices, 0]
#
#     # m == Nf case
#     indices = ((2 * n_range) % Nt) + 1
#     prefactor2s[Nf] = (2 ** (-0.5)) * wave_in[indices, 0]
#
#     # For m = 1, ..., Nf-1
#     m_mid = np.arange(1, Nf)
#     m_grid, n_grid = np.meshgrid(m_mid, n_range, indexing='ij')
#     val = wave_in[n_grid, m_grid]
#     mult2 = np.where(((n_grid + m_grid) % 2) != 0, -1j, 1)
#     prefactor2s[1:Nf] = mult2 * val
#
#     fft_prefactor2s = np.fft.fft(prefactor2s, axis=1)
#
#     res = np.zeros(ND // 2 + 1, dtype=np.complex128)
#
#     # Unpacking for m == 0 and m == Nf
#     for m in [0, Nf]:
#         i_ind_range = np.arange(Nt // 2 + 1 if m == Nf else Nt // 2)
#         i = np.abs(m * Nt // 2 - i_ind_range)
#         ind3 = (2 * i) % Nt
#         res[i] += fft_prefactor2s[m, ind3] * phif[i_ind_range]
#
#     # Unpacking for m = 1,..., Nf-1
#     for m in range(1, Nf):
#         ind31 = (Nt // 2 * m) % Nt
#         ind32 = ind31
#         for i_ind in range(Nt // 2):
#             i1 = Nt // 2 * m - i_ind
#             i2 = Nt // 2 * m + i_ind
#             res[i1] += fft_prefactor2s[m, ind31] * phif[i_ind]
#             res[i2] += fft_prefactor2s[m, ind32] * phif[i_ind]
#             ind31 = (ind31 - 1) % Nt
#             ind32 = (ind32 + 1) % Nt
#         res[Nt // 2 * m] += fft_prefactor2s[m, (Nt // 2 * m) % Nt] * phif[0]
#
#     return res
#
# #
# #
# # if __name__ == '__main__':
# #     phif = np.array(np.random.rand(64))
# #     wave_in = np.array(np.random.rand(64, 64))
# #     Nf = 64
# #     Nt = 64
# #     res = inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)
# #     res2 = inverse_wavelet_freq_helper_fast_version2(wave_in, phif, Nf, Nt)
# #     assert np.allclose(res, res2), "Results do not match!"
