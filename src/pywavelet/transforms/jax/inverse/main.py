import jax.numpy as jnp
from jax.numpy.fft import rfftfreq

from ....types import FrequencySeries, TimeSeries, Wavelet
from ...phi_computer import phi_vec, phitilde_vec_norm
from .to_freq import inverse_wavelet_freq_helper


def from_wavelet_to_time(
    wave_in: Wavelet,
    dt: float,
    nx: float = 4.0,
    mult: int = None,
) -> TimeSeries:
    """Inverse wavelet transform to time domain.

    Parameters
    ----------
    wave_in : Wavelet
        input wavelet
    dt : float
        time step
    nx : float, optional
        parameter for phi_vec, by default 4.0
    mult : int, optional
        parameter for phi_vec, by default 32

    Returns
    -------
    TimeSeries
        Time domain signal
    """
    # Can we just do this?
    freq = from_wavelet_to_freq(wave_in, dt=dt, nx=nx)
    return freq.to_timeseries()


def from_wavelet_to_freq(
    wave_in: Wavelet, dt: float, nx=4.0
) -> FrequencySeries:
    """Inverse wavelet transform to frequency domain.

    Parameters
    ----------
    wave_in : Wavelet
        input wavelet
    dt : float
        time step
    nx : float, optional
        parameter for phitilde_vec_norm, by default 4.0

    Returns
    -------
    FrequencySeries
        Frequency domain signal

    """
    phif = jnp.array(phitilde_vec_norm(wave_in.Nf, wave_in.Nt, d=nx))
    freq_data = inverse_wavelet_freq_helper(
        wave_in.data, phif=phif, Nf=wave_in.Nf, Nt=wave_in.Nt
    )

    freq_data *= 1.0 / jnp.sqrt(2)

    freqs = rfftfreq(wave_in.ND, d=dt)
    return FrequencySeries(data=freq_data, freq=freqs)
