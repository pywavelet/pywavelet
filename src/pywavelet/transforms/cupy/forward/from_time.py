import cupy as cp
from cupyx.scipy.fft import rfft


def transform_wavelet_time_helper(
    data: cp.ndarray, phi: cp.ndarray, Nf: int, Nt: int, mult: int
) -> cp.ndarray:
    """Helper function to do the wavelet transform in the time domain using CuPy"""
    # Define constants
    ND = Nf * Nt
    K = mult * 2 * Nf

    # Pad the data with K extra values
    data_pad = cp.concatenate((data, data[:K]))

    # Generate time bin indices
    time_bins = cp.arange(Nt)
    jj_base = (time_bins[:, None] * Nf - K // 2) % ND
    jj = (jj_base + cp.arange(K)[None, :]) % ND

    # Apply the window (phi) to the data
    wdata = data_pad[jj] * phi[None, :]

    # Perform FFT on the windowed data
    wdata_trans = rfft(wdata, axis=1)

    # Initialize the wavelet transform result
    wave = cp.zeros((Nt, Nf))

    # Handle m=0 case for even time bins
    even_mask = (time_bins % 2 == 0) & (time_bins < Nt - 1)
    even_indices = cp.nonzero(even_mask)[0]

    # Update wave for m=0 using even time bins
    wave[even_indices, 0] = cp.real(wdata_trans[even_indices, 0]) / cp.sqrt(2)
    wave[even_indices + 1, 0] = cp.real(
        wdata_trans[even_indices, Nf * mult]
    ) / cp.sqrt(2)

    # Handle other cases (j > 0) using vectorized operations
    j_range = cp.arange(1, Nf)
    odd_condition = (time_bins[:, None] + j_range[None, :]) % 2 == 1

    wave[:, 1:] = cp.where(
        odd_condition,
        -cp.imag(wdata_trans[:, j_range * mult]),
        cp.real(wdata_trans[:, j_range * mult]),
    )

    return wave.T
