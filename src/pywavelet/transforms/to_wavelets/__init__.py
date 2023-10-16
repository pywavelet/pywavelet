import numpy as np
from ... import fft_funcs as fft
from ..common import phi_vec, phitilde_vec_norm
from .transform_freq_funcs import transform_wavelet_freq_helper
from .transform_time_funcs import transform_wavelet_time_helper
from ...logger import logger

def from_time_to_wavelet(data, Nf, Nt, nx=4., mult=32):
    """From time domain data to wavelet domain

    Warning: there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2

    Parameters
    ----------
    data : array_like
        Time domain data
    Nf : int
        Number of frequency bins
    Nt : int
        Number of time bins
    nx : float, optional
        Number of standard deviations for the gaussian wavelet, by default 4.
    mult : int, optional
        Number of time bins to use for the wavelet transform, by default 32
    """

    if mult > Nt/2:
        logger.warning(f"mult={mult} is too large for Nt={Nt}. This may lead to bogus results.")

    mult = min(mult,Nt//2) #make sure K isn't bigger than ND
    phi = phi_vec(Nf,nx,mult)
    wave = transform_wavelet_time_helper(data,Nf,Nt,phi,mult)

    return wave

def from_time_to_freq_to_wavelet(data, Nf, Nt, nx=4.):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    data_fft = fft.rfft(data)

    return from_freq_to_wavelet(data_fft, Nf, Nt, nx)

def from_freq_to_wavelet(data, Nf, Nt, nx=4.):
    """do the wavelet transform using the fast wavelet domain transform"""
    phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    return transform_wavelet_freq_helper(data,Nf,Nt,phif)
