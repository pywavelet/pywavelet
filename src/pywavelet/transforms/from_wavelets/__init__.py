from typing import Optional

import numpy as np

from ... import fft_funcs as fft
from ...transforms.phi_computer import phi_vec, phitilde_vec_norm
from ..types import FrequencySeries, TimeSeries, Wavelet
from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast


def from_wavelet_to_time(
    wave_in: Wavelet,
    dt: float,
    nx: float = 4.0,
    mult: int = 32,
) -> TimeSeries:
    """fast inverse wavelet transform to time domain"""
    mult = min(mult, wave_in.Nt // 2)  # make sure K isn't bigger than ND
    phi = phi_vec(wave_in.Nf, d=nx, q=mult, dt=dt) / 2
    h_t = inverse_wavelet_time_helper_fast(
        wave_in.data, phi, wave_in.Nf, wave_in.Nt, mult
    )
    ts = np.arange(0, wave_in.Nf * wave_in.Nt) * dt
    return TimeSeries(data=h_t, time=ts)


def from_wavelet_to_freq_to_time(
    wave_in: Wavelet, nx: float = 4.0
) -> TimeSeries:
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = from_wavelet_to_freq(wave_in, nx)
    return fft.irfft(res_f)


def from_wavelet_to_freq(
    wave_in: Wavelet, dt: float, nx=4.0
) -> FrequencySeries:
    """inverse wavelet transform to freq domain signal"""
    phif = phitilde_vec_norm(wave_in.Nf, wave_in.Nt, dt=dt, d=nx)
    freq_data = inverse_wavelet_freq_helper_fast(
        wave_in.data, phif, wave_in.Nf, wave_in.Nt
    )
    freqs = np.fft.fftfreq(wave_in.ND, d=dt)
    return FrequencySeries(data=freq_data, freq=freqs)
