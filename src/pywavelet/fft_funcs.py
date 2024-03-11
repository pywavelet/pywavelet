"""helper to make sure available fft functions are consistent across modules depending on install
mkl-fft is faster so it is the default, but numpy fft is probably more commonly installed to it is the fallback"""
try:
    import mkl_fft

    rfft = mkl_fft.rfft_numpy
    irfft = mkl_fft.irfft_numpy
    fft = mkl_fft.fft
    ifft = mkl_fft.ifft
except ImportError:
    import numpy

    rfft = numpy.fft.rfft
    irfft = numpy.fft.irfft
    fft = numpy.fft.fft
    ifft = numpy.fft.ifft


import numpy as np

from .transforms.types import FrequencySeries, TimeSeries


def periodogram(ts: TimeSeries, wd_func=np.blackman):
    """Compute the periodogram of a time series using the
    Blackman window

    Parameters
    ----------
    ts : ndarray
        intput time series
    wd_func : callable
        tapering window function in the time domain

    Returns
    -------
    ndarray
        periodogram at Fourier frequencies
    """
    fs = ts.sample_rate
    wd = wd_func(ts.data.shape[0])
    k2 = np.sum(wd**2)
    per = np.abs(np.fft.fft(ts.data * wd)) ** 2 * 2 / (k2 * fs)
    freq = np.fft.fftfreq(len(ts)) * fs
    return FrequencySeries(data=per, freq=freq)
