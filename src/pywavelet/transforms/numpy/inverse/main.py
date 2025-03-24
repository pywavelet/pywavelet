import numpy as np

from ....types import FrequencySeries, TimeSeries, Wavelet
from ...phi_computer import phi_vec, phitilde_vec_norm
from .to_freq import inverse_wavelet_freq_helper_fast
from .to_time import inverse_wavelet_time_helper_fast

__all__ = ["from_wavelet_to_time", "from_wavelet_to_freq"]


def from_wavelet_to_time(
    wave_in: Wavelet,
    dt: float,
    nx: float = 4.0,
    mult: int = 32,
) -> TimeSeries:
    """
    Perform an inverse wavelet transform to the time domain.

    This function converts a wavelet-domain signal to a time-domain signal using
    the inverse wavelet transform algorithm.

    Parameters
    ----------
    wave_in : Wavelet
        Input wavelet, represented by a `Wavelet` object.
    dt : float
        Time step of the wavelet data.
    nx : float, optional
        Scaling parameter for the phi vector used in the transformation. Default is 4.0.
    mult : int, optional
        Multiplier parameter for the phi vector. Ensures that the `mult` value
        is not larger than half the number of time bins (`wave_in.Nt`). Default is 32.

    Returns
    -------
    TimeSeries
        A `TimeSeries` object containing the signal transformed into the time domain.

    Notes
    -----
    The transformation involves normalizing the output by the square root of 2
    to ensure the proper backwards transformation.
    """

    mult = min(mult, wave_in.Nt // 2)  # Ensure mult is not larger than ND/2
    phi = phi_vec(wave_in.Nf, d=nx, q=mult) / 2
    h_t = inverse_wavelet_time_helper_fast(
        wave_in.data.T, phi, wave_in.Nf, wave_in.Nt, mult
    )
    h_t *= 1.0 / np.sqrt(2)  # Normalize to get proper backward transformation
    ts = np.arange(0, wave_in.Nf * wave_in.Nt) * dt
    return TimeSeries(data=h_t, time=ts)


def from_wavelet_to_freq(
    wave_in: Wavelet, dt: float, nx: float = 4.0
) -> FrequencySeries:
    """
    Perform an inverse wavelet transform to the frequency domain.

    This function converts a wavelet-domain signal into a frequency-domain
    signal using the inverse wavelet transform algorithm.

    Parameters
    ----------
    wave_in : Wavelet
        Input wavelet, represented by a `Wavelet` object.
    dt : float
        Time step of the wavelet data.
    nx : float, optional
        Scaling parameter for the phi vector used in the transformation. Default is 4.0.

    Returns
    -------
    FrequencySeries
        A `FrequencySeries` object containing the signal transformed into the frequency domain.

    Notes
    -----
    The transformation involves normalizing the output by the square root of 2
    to ensure the proper backwards transformation.
    """

    phif = phitilde_vec_norm(wave_in.Nf, wave_in.Nt, d=nx)
    freq_data = inverse_wavelet_freq_helper_fast(
        wave_in.data, phif, wave_in.Nf, wave_in.Nt
    )

    freq_data *= 1.0 / np.sqrt(2)

    freqs = np.fft.rfftfreq(wave_in.ND, d=dt)
    return FrequencySeries(data=freq_data, freq=freqs)
