"""helper functions for transform_time.py"""
import numpy as np
from . import fft_funcs as fft
from .transform_freq_funcs import phitilde_vec_norm,transform_wavelet_freq_helper
from .transform_time_funcs import phi_vec,transform_wavelet_time_helper
from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast

def inverse_wavelet_time(wave_in,Nf,Nt,nx=4.,mult=32):
    """fast inverse wavelet transform to time domain"""
    mult = min(mult,Nt//2) #make sure K isn't bigger than ND
    phi = phi_vec(Nf,nx=nx,mult=mult)/2

    return inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,mult)

def inverse_wavelet_freq_time(wave_in,Nf,Nt,nx=4.):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    res_f = inverse_wavelet_freq(wave_in,Nf,Nt,nx)
    return fft.irfft(res_f)


def inverse_wavelet_freq(wave_in,Nf,Nt,nx=4.):
    """inverse wavelet transform to freq domain signal"""
    phif = phitilde_vec_norm(Nf,Nt,nx)
    return inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)

def transform_wavelet_time(data,Nf,Nt,nx=4.,mult=32):
    """do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2"""
    mult = min(mult,Nt//2) #make sure K isn't bigger than ND
    phi = phi_vec(Nf,nx,mult)
    wave = transform_wavelet_time_helper(data,Nf,Nt,phi,mult)

    return wave

def transform_wavelet_freq_time(data,Nf,Nt,nx=4.):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    data_fft = fft.rfft(data)

    return transform_wavelet_freq(data_fft,Nf,Nt,nx)

def transform_wavelet_freq(data,Nf,Nt,nx=4.):
    """do the wavelet transform using the fast wavelet domain transform"""
    phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    return transform_wavelet_freq_helper(data,Nf,Nt,phif)
