from pywavelet import fft_funcs as fft
from pywavelet.transforms.common import phi_vec, phitilde_vec_norm

from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast


def from_wavelet_to_time(wave_in, Nf, Nt, nx=4.0, mult=32):
    """fast inverse wavelet transform to time domain"""
    mult = min(mult, Nt // 2)  # make sure K isn't bigger than ND
    phi = phi_vec(Nf, nx=nx, mult=mult) / 2

    return inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)


def from_wavelet_to_freq_to_time(wave_in, Nf, Nt, nx=4.0):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = from_wavelet_to_freq(wave_in, Nf, Nt, nx)
    return fft.irfft(res_f)


def from_wavelet_to_freq(wave_in, Nf, Nt, nx=4.0):
    """inverse wavelet transform to freq domain signal"""
    phif = phitilde_vec_norm(Nf, Nt, nx)
    return inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)
