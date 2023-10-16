"""helper functions for transform_time.py"""
import numpy as np
from numba import njit
from ... import fft_funcs as fft


def transform_wavelet_time_helper(data,Nf:int,Nt:int,phi,mult:int)->np.ndarray:
    """helper function to do the wavelet transform in the time domain"""
    # the time domain data stream
    ND = Nf*Nt

    K = mult*2*Nf

    assert len(data) == ND, f"len(data)={len(data)} != Nf*Nt={ND}"

    # windowed data packets
    wdata = np.zeros(K)

    wave = np.zeros((Nt,Nf))  # wavelet wavepacket transform of the signal
    data_pad = np.zeros(ND+K)
    data_pad[:ND] = data
    data_pad[ND:ND+K] = data[:K]

    for i in range(0,Nt):
        assign_wdata(i,K,ND,Nf,wdata,data_pad,phi)
        wdata_trans = fft.rfft(wdata,K)
        pack_wave(i,mult,Nf,wdata_trans,wave)

    return wave

@njit()
def assign_wdata(i,K,ND,Nf,wdata,data_pad,phi):
    """assign wdata to be fftd in loop, data_pad needs K extra values on the right to loop"""
    #half_K = np.int64(K/2)
    jj = i*Nf-K//2
    if jj<0:
        jj += ND  # periodically wrap the data
    if jj>=ND:
        jj -= ND # periodically wrap the data
    for j in range(0,K):
        #jj = i*Nf-half_K+j
        wdata[j] = data_pad[jj]*phi[j]  # apply the window
        jj += 1
        #if jj==ND:
        #    jj -= ND # periodically wrap the data

@njit()
def pack_wave(i,mult,Nf,wdata_trans,wave):
    """pack fftd wdata into wave array"""
    if i%2==0 and i<wave.shape[0]-1:
        #m=0 value at even Nt and
        wave[i,0] = np.real(wdata_trans[0])/np.sqrt(2)
        wave[i+1,0] = np.real(wdata_trans[Nf*mult])/np.sqrt(2)

    for j in range(1,Nf):
        if (i+j)%2:
            wave[i,j] = -np.imag(wdata_trans[j*mult])
        else:
            wave[i,j] = np.real(wdata_trans[j*mult])

