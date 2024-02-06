from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast
from ..types import Wavelet, TimeSeries
from ... import fft_funcs as fft
from ...transforms.common import phi_vec, phitilde_vec_norm

from typing import Optional
import numpy as np


def from_wavelet_to_time(wave_in: Wavelet,  nx: float = 4.0, mult: int = 32,
                         dt: Optional[float] = 1) -> TimeSeries:
    """fast inverse wavelet transform to time domain"""
    mult = min(mult, wave_in.Nt // 2)  # make sure K isn't bigger than ND
    phi = phi_vec(wave_in.Nf, nx=nx, mult=mult) / 2
    h_t = inverse_wavelet_time_helper_fast(wave_in.data, phi, wave_in.Nf, wave_in.Nt, mult)
    ts = np.arange(0, wave_in.Nf * wave_in.Nt) * dt
    return TimeSeries(data=h_t, time=ts)


def from_wavelet_to_freq_to_time(wave_in: Wavelet, Nf: int, Nt: int, nx: float = 4.0) -> TimeSeries:
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = from_wavelet_to_freq(wave_in, Nf, Nt, nx)
    return fft.irfft(res_f)


def from_wavelet_to_freq(wave_in, Nf, Nt, nx=4.0):
    """inverse wavelet transform to freq domain signal"""
    phif = phitilde_vec_norm(Nf, Nt, nx)
    return inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)
