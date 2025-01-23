import jax.numpy as jnp
from jax.numpy.fft import rfftfreq

from ...phi_computer import phi_vec, phitilde_vec_norm
from ....types import FrequencySeries, TimeSeries, Wavelet
from .to_freq import inverse_wavelet_freq_helper
# from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper


def from_wavelet_to_time(
    wave_in: Wavelet,
    dt: float,
    nx: float = 4.0,
    mult: int = 32,
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
    phif = jnp.array(phitilde_vec_norm(wave_in.Nf, wave_in.Nt, dt=dt, d=nx))
    freq_data = inverse_wavelet_freq_helper(
        wave_in.data, phif=phif, Nf=wave_in.Nf, Nt=wave_in.Nt
    )

    freq_data *= 2 ** (
        -1 / 2
    )  # Normalise to get the proper backwards transformation

    freqs = rfftfreq(wave_in.ND*2, d=dt)[1:]
    return FrequencySeries(data=freq_data, freq=freqs)