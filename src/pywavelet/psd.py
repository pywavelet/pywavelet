from typing import Optional, Tuple, Union

import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .transforms import from_time_to_wavelet
from .transforms.types import (
    FrequencySeries,
    TimeSeries,
    Wavelet,
    wavelet_dataset,
)

DATA_TYPE = Union[TimeSeries, FrequencySeries, Wavelet]


def evolutionary_psd_from_stationary_psd(
        psd: np.ndarray,
        psd_f: np.ndarray,
        f_grid: np.ndarray,
        t_grid: np.ndarray,
        dt: float,
) -> Wavelet:
    """
    PSD[ti,fi] = PSD[fi] / dt
    """
    Nt = len(t_grid)
    delta_f = f_grid[1] - f_grid[0]
    nan_val = np.max(psd)
    psd_grid = interp1d(
        psd_f,
        psd,
        kind="nearest",
        fill_value=nan_val,
        bounds_error=False,
    )(f_grid)

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0)/dt  
    w = wavelet_dataset(psd_grid, time_grid=t_grid, freq_grid=f_grid)
    return w


def generate_noise_from_psd(
        psd_func,
        n_data,
        fs,
        noise_type: Optional[DATA_TYPE] = FrequencySeries,
) -> Union[TimeSeries, FrequencySeries, Wavelet]:
    """
    Noise generator from arbitrary power spectral density.
    Uses a Gaussian random generation in the frequency domain.

    Parameters
    ----------
    psd_func: callable
        one-sided PSD function in A^2 / Hz, where A is the unit of the desired
        output time series. Can also return a p x p spectrum matrix
    n_data: int
        size of output time series
    fs: float
        sampling frequency in Hz


    Returns
    -------
    tseries: ndarray
        generated time series

    """

    tmax = n_data / fs
    # Number of points to generate in the frequency domain (circulant embedding)
    n_psd = 2 * n_data
    # Number of positive frequencies
    n_fft = int((n_psd - 1) / 2)
    # Frequency array
    f = np.fft.fftfreq(n_psd) * fs
    # Avoid zero frequency as it sometimes makes the PSD infinite
    f[0] = f[1]
    # Compute the PSD (or the spectrum matrix)
    psd_f = psd_func(np.abs(f))

    if psd_f.ndim == 1:
        psd_sqrt = np.sqrt(psd_f)
        # Real part of the Noise fft : it is a gaussian random variable
        noise_tf_real = (
                np.sqrt(0.5)
                * psd_sqrt[0: n_fft + 1]
                * np.random.normal(loc=0.0, scale=1.0, size=n_fft + 1)
        )
        # Imaginary part of the Noise fft :
        noise_tf_im = (
                np.sqrt(0.5)
                * psd_sqrt[0: n_fft + 1]
                * np.random.normal(loc=0.0, scale=1.0, size=n_fft + 1)
        )
        # The Fourier transform must be real in f = 0
        noise_tf_im[0] = 0.0
        noise_tf_real[0] = noise_tf_real[0] * np.sqrt(2.0)
        # Create the NoiseTF complex numbers for positive frequencies
        noise_tf = noise_tf_real + 1j * noise_tf_im

    # To get a real valued signal we must have NoiseTF(-f) = NoiseTF*
    if (n_psd % 2 == 0) & (psd_f.ndim == 1):
        # The TF at Nyquist frequency must be real in the case of an even
        # number of data
        noise_sym0 = np.array([psd_sqrt[n_fft + 1] * np.random.normal(0, 1)])
        # Add the symmetric part corresponding to negative frequencies
        noise_tf = np.hstack(
            (noise_tf, noise_sym0, np.conj(noise_tf[1: n_fft + 1])[::-1])
        )
    elif (n_psd % 2 != 0) & (psd_f.ndim == 1):
        noise_tf = np.hstack(
            (noise_tf, np.conj(noise_tf[1: n_fft + 1])[::-1])
        )

    if noise_type == FrequencySeries:
        return FrequencySeries(data=noise_tf, freq=f)
    elif noise_type == TimeSeries:
        tseries = np.fft.ifft(np.sqrt(n_psd * fs / 2.0) * noise_tf, axis=0)
        delta_t = 1 / fs  # Sampling interval -- largely oversampling here.
        n_data = 2 ** int(np.log(tmax / delta_t) / np.log(2))
        t = np.arange(0, n_data) * delta_t
        return TimeSeries(data=tseries[0:n_data].real, time=t)
