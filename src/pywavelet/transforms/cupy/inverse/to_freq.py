import cupy as cp
from cupyx.scipy.fft import fft


def inverse_wavelet_freq_helper(
    wave_in: cp.ndarray, phif: cp.ndarray, Nf: int, Nt: int
) -> cp.ndarray:
    """CuPy vectorized function for inverse_wavelet_freq"""
    wave = wave_in.T
    ND2 = (Nf * Nt) // 2
    half = Nt // 2

    # === STEP 1: build prefactor2s[m, n] ===
    m_range = cp.arange(Nf + 1)
    pref2 = cp.zeros((Nf + 1, Nt), dtype=cp.complex128)
    n = cp.arange(Nt)

    # m=0 and m=Nf, with 1/√2
    pref2[0, :] = wave[(2 * n) % Nt, 0] * (2 ** (-0.5))
    pref2[Nf, :] = wave[((2 * n) % Nt) + 1, 0] * (2 ** (-0.5))

    # middle m=1...Nf-1
    m_mid = cp.arange(1, Nf)
    # build meshgrids (m_mid rows, n cols)
    mm, nn = cp.meshgrid(m_mid, n, indexing="ij")
    vals = wave[nn, mm]
    signs = cp.where(((nn + mm) % 2) == 1, -1j, 1)
    pref2[1:Nf, :] = signs * vals

    # === STEP 2: FFT along time axis ===
    F = fft(pref2, axis=1)  # shape (Nf+1, Nt)

    # === STEP 3: unpack back into half-spectrum res[0...ND2] ===
    res = cp.zeros(ND2 + 1, dtype=cp.complex128)
    idx = cp.arange(half)

    # 3a) contribution from m=0
    res[idx] += F[0, (2 * idx) % Nt] * phif[idx]

    # 3b) contribution from m=Nf
    iNf = cp.abs(Nf * half - idx)
    res[iNf] += F[Nf, (2 * idx) % Nt] * phif[idx]

    # special Nyquist‐folding term
    special = cp.abs(Nf * half - half)
    res[special] += F[Nf, 0] * phif[half]

    # 3c) middle m cases
    m_mid = cp.arange(1, Nf)
    i_mid = cp.arange(half)  # 0...half-1
    mm, ii = cp.meshgrid(m_mid, i_mid, indexing="ij")
    i1 = (half * mm - ii) % (ND2 + 1)
    i2 = (half * mm + ii) % (ND2 + 1)
    ind1 = (half * mm - ii) % Nt
    ind2 = (half * mm + ii) % Nt

    # accumulate
    res[i1] += F[mm, ind1] * phif[ii]
    res[i2] += F[mm, ind2] * phif[ii]

    # override the "center" points (j=0) exactly
    centers = half * m_mid
    fft_idx = centers % Nt
    res[centers] = F[m_mid, fft_idx] * phif[0]

    return res
