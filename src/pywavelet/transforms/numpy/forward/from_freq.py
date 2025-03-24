"""helper functions for transform_freq"""

import logging

import numpy as np
from numba import njit

# import rocket_fft.special as rfft  # JIT‐able FFT routines


logger = logging.getLogger("pywavelet")


def transform_wavelet_freq_helper(
    data: np.ndarray, Nf: int, Nt: int, phif: np.ndarray
) -> np.ndarray:
    """
    Forward wavelet transform helper using the fast wavelet domain transform,
    with a JIT-able FFT (rocket-fft) so that the whole transform is jittable.

    Parameters
    ----------
    data : np.ndarray
        Input frequency-domain data (1D array).
    Nf : int
        Number of frequency bins.
    Nt : int
        Number of time bins.
    phif : np.ndarray
        Fourier-domain phase factors (complex-valued array of length Nt//2 + 1).

    Returns
    -------
    wave : np.ndarray
        The wavelet transform output of shape (Nt, Nf). Note that contributions from
        f_bin==0 and f_bin==Nf are both stored in column 0.
    """
    logger.debug(
        f"[NUMPY TRANSFORM] Input types [data:{type(data)}, phif:{type(phif)}]"
    )
    wave = np.zeros((Nt, Nf), dtype=np.float64)
    DX = np.zeros(Nt, dtype=np.complex128)
    # Create a copy of the input data (if needed).
    freq_strain = data.copy()
    __core(Nf, Nt, DX, freq_strain, phif, wave)
    return wave


@njit()
def __core(
    Nf: int,
    Nt: int,
    DX: np.ndarray,
    data: np.ndarray,
    phif: np.ndarray,
    wave: np.ndarray,
):
    """
    Process each frequency bin (f_bin) to compute the temporary array DX,
    perform the inverse FFT using rocket-fft, and then unpack the result into wave.

    This function is fully jittable.
    """
    for f_bin in range(0, Nf + 1):
        __fill_wave_1(f_bin, Nt, Nf, DX, data, phif)
        # Use rocket-fft's ifft (which is JIT-able) instead of np.fft.ifft.
        DX_trans = np.fft.ifft(DX, Nt)
        __fill_wave_2(f_bin, DX_trans, wave, Nt, Nf)


@njit()
def __fill_wave_1(
    f_bin: int,
    Nt: int,
    Nf: int,
    DX: np.ndarray,
    data: np.ndarray,
    phif: np.ndarray,
) -> None:
    """
    Fill the temporary complex array DX for the given frequency bin (f_bin)
    based on the input data and the phase factors phif.

    The computation is performed over a window of indices defined by the current f_bin.
    """
    i_base = Nt // 2
    jj_base = f_bin * (Nt // 2)

    # Special center assignment:
    if f_bin == 0 or f_bin == Nf:
        DX[i_base] = phif[0] * data[jj_base] / 2.0
    else:
        DX[i_base] = phif[0] * data[jj_base]

    # Determine the window of indices.
    start = jj_base + 1 - (Nt // 2)
    end = jj_base + (Nt // 2)
    for jj in range(start, end):
        j = np.abs(jj - jj_base)
        i = i_base - jj_base + jj
        # For the highest frequency (f_bin==Nf) or the lowest (f_bin==0), zero out the out-of-range values.
        if (f_bin == Nf and jj > jj_base) or (f_bin == 0 and jj < jj_base):
            DX[i] = 0.0
        elif j == 0:
            # Center already assigned.
            continue
        else:
            DX[i] = phif[j] * data[jj]


@njit()
def __fill_wave_2(
    f_bin: int, DX_trans: np.ndarray, wave: np.ndarray, Nt: int, Nf: int
) -> None:
    """
    Unpack the inverse FFT output (DX_trans) into the output wave array.

    For f_bin==0 and f_bin==Nf, the results are stored in column 0 of wave,
    using even- or odd-indexed rows respectively. For intermediate f_bin values,
    the values are stored in column f_bin with a sign and component (real or imag)
    determined by parity.
    """
    sqrt2 = np.sqrt(2.0)
    if f_bin == 0:
        # f_bin==0: assign even-indexed rows of column 0.
        for n in range(0, Nt, 2):
            wave[n, 0] = DX_trans[n].real * sqrt2
    elif f_bin == Nf:
        # f_bin==Nf: assign odd-indexed rows of column 0.
        for n in range(0, Nt, 2):
            wave[n + 1, 0] = DX_trans[n].real * sqrt2
    else:
        # For intermediate f_bin, assign values to column f_bin.
        for n in range(0, Nt):
            if f_bin % 2:
                # For odd f_bin: use -imag when (n+f_bin) is odd; otherwise use real.
                if (n + f_bin) % 2:
                    wave[n, f_bin] = -DX_trans[n].imag
                else:
                    wave[n, f_bin] = DX_trans[n].real
            else:
                # For even f_bin: use imag when (n+f_bin) is odd; otherwise use real.
                if (n + f_bin) % 2:
                    wave[n, f_bin] = DX_trans[n].imag
                else:
                    wave[n, f_bin] = DX_trans[n].real
