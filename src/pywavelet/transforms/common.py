import numpy as np
import scipy
from .. import fft_funcs as fft


def phitilde_vec(om,Nf,nx=4.):
    """compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    OM = np.pi  #Nyquist angular frequency
    DOM = OM/Nf #2 pi times DF
    insDOM = 1./np.sqrt(DOM)
    B = OM/(2*Nf)
    A = (DOM-B)/2
    z = np.zeros(om.size)

    mask = (np.abs(om)>= A)&(np.abs(om)<A+B)

    x = (np.abs(om[mask])-A)/B
    y = scipy.special.betainc(nx,nx, x)
    z[mask] = insDOM*np.cos(np.pi/2.*y)

    z[np.abs(om)<A] = insDOM
    return z

def phitilde_vec_norm(Nf,Nt,nx):
    """normalize phitilde as needed for inverse frequency domain transform"""
    ND = Nf*Nt
    oms = 2*np.pi/ND*np.arange(0,Nt//2+1)
    phif = phitilde_vec(oms,Nf,nx)
    #nrm should be 1
    nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/ND)
    nrm /= np.pi**(3/2)/np.pi
    phif /= nrm
    return phif



def phi_vec(Nf,nx=4.,mult=16):
    """get time domain phi as fourier transform of phitilde_vec"""
    #TODO fix mult

    OM = np.pi
    DOM = OM/Nf
    insDOM = 1./np.sqrt(DOM)
    K = mult*2*Nf
    half_K = mult*Nf#np.int64(K/2)

    dom = 2*np.pi/K  # max frequency is K/2*dom = pi/dt = OM

    DX = np.zeros(K,dtype=np.complex128)

    #zero frequency
    DX[0] =  insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1:half_K+1] = phitilde_vec(dom*np.arange(1,half_K+1),Nf,nx)
    # negative frequencies
    DX[half_K+1:] = phitilde_vec(-dom*np.arange(half_K-1,0,-1),Nf,nx)
    DX = K*fft.ifft(DX,K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(K/dom)#*np.linalg.norm(phi)

    fac = np.sqrt(2.0)/nrm
    phi *= fac
    return phi
