import numpy as np

from ...transforms.phi_computer import phi_vec, phitilde_vec_norm
from ..types import FrequencySeries, TimeSeries, Wavelet
from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper


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
    wave_in = wave_in.T
    mult = min(mult, wave_in.Nt // 2)  # make sure K isn't bigger than ND
    phi = phi_vec(wave_in.Nf, d=nx, q=mult, dt=dt) / 2
    h_t = inverse_wavelet_time_helper(
        wave_in.data, phi, wave_in.Nf, wave_in.Nt, mult
    )
    h_t *= 2 ** -(
        1 / 2
    )  # We must normalise by this to get proper backwards transformation

    ts = np.arange(0, wave_in.Nf * wave_in.Nt) * dt
    return TimeSeries(data=h_t, time=ts)


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
    phif = phitilde_vec_norm(wave_in.Nf, wave_in.Nt, dt=dt, d=nx)
    freq_data = inverse_wavelet_freq_helper(
        wave_in.data, phif, wave_in.Nf, wave_in.Nt
    )

    freq_data *= 2 ** (
        -1 / 2
    )  # Normalise to get the proper backwards transformation

    freqs = np.fft.rfftfreq(wave_in.ND, d=dt)
    return FrequencySeries(data=freq_data, freq=freqs)
