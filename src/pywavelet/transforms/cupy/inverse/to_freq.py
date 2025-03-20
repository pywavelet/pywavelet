import cupy as cp
from cupyx.scipy.fft import fft


def inverse_wavelet_freq_helper(
    wave_in: cp.ndarray, phif: cp.ndarray, Nf: int, Nt: int
) -> cp.ndarray:
    """CuPy vectorized function for inverse_wavelet_freq"""
    wave_in = wave_in.T
    ND = Nf * Nt

    m_range = cp.arange(Nf + 1)
    prefactor2s = cp.zeros((Nf + 1, Nt), dtype=cp.complex128)

    n_range = cp.arange(Nt)

    # m == 0 case
    prefactor2s[0] = 2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt, 0]

    # m == Nf case
    prefactor2s[Nf] = 2 ** (-1 / 2) * wave_in[(2 * n_range) % Nt + 1, 0]

    # Other m cases
    m_mid = m_range[1:Nf]
    n_grid, m_grid = cp.meshgrid(n_range, m_mid)
    val = wave_in[n_grid, m_grid]
    mult2 = cp.where((n_grid + m_grid) % 2, -1j, 1)
    prefactor2s[1:Nf] = mult2 * val

    # Vectorized FFT
    fft_prefactor2s = fft(prefactor2s, axis=1)

    # Vectorized __unpack_wave_inverse
    res = cp.zeros(ND, dtype=cp.complex128)

    # m == 0 or m == Nf cases
    i_ind_range = cp.arange(Nt // 2)
    i_0 = cp.abs(i_ind_range)
    i_Nf = cp.abs(Nf * Nt // 2 - i_ind_range)
    ind3_0 = (2 * i_0) % Nt
    ind3_Nf = (2 * i_Nf) % Nt

    res[i_0] += fft_prefactor2s[0, ind3_0] * phif[i_ind_range]
    res[i_Nf] += fft_prefactor2s[Nf, ind3_Nf] * phif[i_ind_range]

    # Special case for m == Nf
    res[Nf * Nt // 2] += fft_prefactor2s[Nf, 0] * phif[Nt // 2]

    # Other m cases
    m_mid = m_range[1:Nf]
    i_ind_range = cp.arange(Nt // 2 + 1)
    m_grid, i_ind_grid = cp.meshgrid(m_mid, i_ind_range)

    i1 = Nt // 2 * m_grid - i_ind_grid
    i2 = Nt // 2 * m_grid + i_ind_grid
    ind31 = (Nt // 2 * m_grid - i_ind_grid) % Nt
    ind32 = (Nt // 2 * m_grid + i_ind_grid) % Nt

    res[i1] += fft_prefactor2s[m_grid, ind31] * phif[i_ind_grid]
    res[i2] += fft_prefactor2s[m_grid, ind32] * phif[i_ind_grid]

    return res
