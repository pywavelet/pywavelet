"""functions for computing the inverse wavelet transforms"""
from numba import njit
import numpy as np

from . import fft_funcs as fft

def inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,mult):
    """helper loop for fast inverse wavelet transform"""
    ND = Nf*Nt
    K = mult*2*Nf
    #res = np.zeros(ND)

    #extend this array, we can use wrapping boundary conditions at end
    res = np.zeros(ND+K+Nf)

    afins = np.zeros(2*Nf,dtype=np.complex128)

    for n in range(0,Nt):
        if n%2==0:
            pack_wave_time_helper_compact(n,Nf,Nt,wave_in,afins)
            ffts_fin = fft.fft(afins)
            unpack_time_wave_helper_compact(n,Nf,Nt,K,phi,ffts_fin,res)

    #wrap boundary conditions
    res[:min(K+Nf,ND)] += res[ND:min(ND+K+Nf,2*ND)]
    if K+Nf>ND:
        res[:K+Nf-ND] += res[2*ND:ND+K*Nf]

    res = res[:ND]

    return res

@njit()
def unpack_time_wave_helper(n,Nf,Nt,K,phis,fft_fin_real,res):
    """helper for time domain wavelet transform to unpack wavelet domain coefficients"""
    ND = Nf*Nt

    idxf = (-K//2+n*Nf+ND)%(2*Nf)
    k = (-K//2+n*Nf)%ND

    for k_ind in range(0,K):
        res_loc = fft_fin_real[idxf]
        res[k] += phis[k_ind]*res_loc
        idxf+=1
        k += 1

        if idxf==2*Nf:
            idxf = 0
        if k==ND:
            k = 0

@njit()
def unpack_time_wave_helper_compact(n,Nf,Nt,K,phis,fft_fin,res):
    """helper for time domain wavelet transform to unpack wavelet domain coefficients
    in compact representation where cosine and sine parts are real and imaginary parts"""
    ND = Nf*Nt
    fft_fin_real = np.zeros(4*Nf)
    fft_fin_imag = np.zeros(4*Nf)
    for itrf in range(0,2*Nf):
        fft_fin_real[itrf] = np.real(fft_fin[itrf])
        fft_fin_real[itrf+2*Nf] = fft_fin_real[itrf]
        fft_fin_imag[itrf] = np.imag(fft_fin[(itrf+Nf)%(2*Nf)])
        fft_fin_imag[itrf+2*Nf] = fft_fin_imag[itrf]

    idxf1_base = (-K//2+n*Nf+ND)%(2*Nf)
    k1_base = (-K//2+n*Nf)%ND
    for k_ind in range(0,K,2*Nf):
        for idxf1_add in range(0,2*Nf):
            idxf1 = idxf1_base+idxf1_add#k_ind%(2*Nf)
            k_ind_loc = k_ind+idxf1_add
            k1 = k1_base+k_ind_loc

            res[k1] += phis[k_ind_loc]*fft_fin_real[idxf1]
            res[k1+Nf] += phis[k_ind_loc]*fft_fin_imag[idxf1]


@njit()
def pack_wave_time_helper(n,Nf,Nt,wave_in,afins):
    """helper for time domain transform to pack wavelet domain coefficients"""
    if n%2==0:
        #assign highest and lowest bin correctly
        afins[0] = np.sqrt(2)*wave_in[n,0]
        if n+1<Nt:
            afins[Nf] = np.sqrt(2)*wave_in[n+1,0]
    else:
        afins[0] = 0.
        afins[Nf] = 0.

    for idxm in range(0,Nf//2-1):
        if n%2:
            afins[2*idxm+2] = 1j*wave_in[n,2*idxm+2]
            afins[2*Nf-2*idxm-2] = -1j*wave_in[n,2*idxm+2]
        else:
            afins[2*idxm+2] = 1*wave_in[n,2*idxm+2]
            afins[2*Nf-2*idxm-2] = 1*wave_in[n,2*idxm+2]

    for idxm in range(0,Nf//2):
        if n%2:
            afins[2*idxm+1] = -1*wave_in[n,2*idxm+1]
            afins[2*Nf-2*idxm-1] = -1*wave_in[n,2*idxm+1]
        else:
            afins[2*idxm+1] = 1j*wave_in[n,2*idxm+1]
            afins[2*Nf-2*idxm-1] = -1j*wave_in[n,2*idxm+1]

@njit()
def pack_wave_time_helper_compact(n,Nf,Nt,wave_in,afins):
    """helper for time domain transform to pack wavelet domain coefficients
    in packed representation with odd and even coefficients in real and imaginary pars"""
    afins[0] = np.sqrt(2)*wave_in[n,0]
    if n+1<Nt:
        afins[Nf] = np.sqrt(2)*wave_in[n+1,0]

    for idxm in range(0,Nf-2,2):
        afins[idxm+2] = wave_in[n,idxm+2]-wave_in[n+1,idxm+2]
        afins[2*Nf-idxm-2] = wave_in[n,idxm+2]+wave_in[n+1,idxm+2]

        afins[idxm+1] = 1j*(wave_in[n,idxm+1]-wave_in[n+1,idxm+1])
        afins[2*Nf-idxm-1] = -1j*(wave_in[n,idxm+1]+wave_in[n+1,idxm+1])

    afins[Nf-1] = 1j*(wave_in[n,Nf-1]-wave_in[n+1,Nf-1])
    afins[Nf+1] = -1j*(wave_in[n,Nf-1]+wave_in[n+1,Nf-1])
