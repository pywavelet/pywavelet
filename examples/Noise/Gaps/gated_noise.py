import os

import numpy as np
from corner import corner
from scipy.signal.windows import tukey
from tqdm import tqdm

from numpy.random import normal
import pytest

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries,TimeSeries
from pywavelet.utils.lisa import get_lisa_data, waveform, FFT, zero_pad

from pywavelet.utils.lvk import inject_signal_in_noise
from pywavelet.utils.snr import compute_snr
from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_freq, from_wavelet_to_time

from gap_funcs import gap_routine, get_Cov
np.random.seed(1234)



def PowerSpectralDensity(f):
    """
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf
    Removed galactic confusion noise. Non stationary effect.
    """

    L = 2.5 * 10**9  # Length of LISA arm
    f0 = 19.09 * 10**-3

    Poms = ((1.5 * 10**-11) ** 2) * (
        1 + ((2 * 10**-3) / f) ** 4
    )  # Optical Metrology Sensor
    Pacc = (
        (3 * 10**-15) ** 2
        * (1 + (4 * 10**-3 / (10 * f)) ** 2)
        * (1 + (f / (8 * 10**-3)) ** 4)
    )  # Acceleration Noise

    PSD = (
        (10 / (3 * L**2))
        * (Poms + (4 * Pacc) / ((2 * np.pi * f)) ** 4)
        * (1 + 0.6 * (f / f0) ** 2)
    )  # PSD

    return PSD


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def FFT(waveform):
    """
    Here we taper the signal, pad and then compute the FFT. We remove the zeroth frequency bin because
    the PSD (for which the frequency domain waveform is used with) is undefined at f = 0.
    """
    N = len(waveform)
    taper = tukey(N, 0.1)
    waveform_w_pad = __zero_pad(waveform * taper)
    return np.fft.rfft(waveform_w_pad)


def freq_PSD(waveform_t, delta_t):
    """
    Here we take in a waveform and sample the correct fourier frequencies and output the PSD. There is no
    f = 0 frequency bin because the PSD is undefined there.
    """
    n_t = len(__zero_pad(waveform_t))
    freq = np.fft.rfftfreq(n_t, delta_t)
    freq[0] = freq[1] # redefining zeroth frequency to stop PSD -> infinity
    PSD = PowerSpectralDensity(freq)

    return freq, PSD


def inner_prod(sig1_f, sig2_f, PSD, delta_t, N_t):
    # Compute inner product. Useful for likelihood calculations and SNRs.
    return (4 * delta_t / N_t) * np.real(
        sum(np.conjugate(sig1_f) * sig2_f / PSD)
    )


def waveform(a, f, fdot, t, eps=0):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))


def llike(data_wavelet, signal_wavelet, psd_wavelet):
    """
    Computes log likelihood
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain
    Proposed signal in frequency domain
    Variance of noise
    """
    inn_prod_wavelet = np.nansum(((data_wavelet - signal_wavelet) ** 2) / psd_wavelet)
    return -0.5 * inn_prod_wavelet

# Set true parameters. These are the parameters we want to estimate using MCMC.

a_true = 1e-20
f_true = 3e-3
fdot_true = 1e-8

tmax = 10 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(
    0.4 / fs
)  # Sampling interval -- largely oversampling here.

t = np.arange(
    0, tmax, delta_t
)  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N = int(
    2 ** (np.ceil(np.log2(len(t))))
)  # Round length of time series to a power of two.
# Length of time series
h_t = waveform(a_true, f_true, fdot_true, t)
h_t_pad = zero_pad(h_t)

t_pad = np.arange(0,len(h_t_pad)*delta_t, delta_t)

h_true_f = np.fft.rfft(h_t_pad)
freq, PSD = freq_PSD(t, delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))



# Compute things in the wavelet domain

signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
psd = FrequencySeries(data = PSD, freq = freq)


kwgs = dict(
    Nf=512,
)


h_wavelet = Data.from_frequencyseries(signal_f_series, **kwgs).wavelet
psd_wavelet = evolutionary_psd_from_stationary_psd(
                                                    psd=psd.data,
                                                    psd_f=psd.freq,
                                                    f_grid=h_wavelet.freq,
                                                    t_grid=h_wavelet.time,
                                                    dt=delta_t,
                                                )

SNR2_wavelet = np.nansum((h_wavelet*h_wavelet) / psd_wavelet)
breakpoint()
print("SNR in wavelet domain is", SNR2_wavelet**(1/2))

# Gaps in the frequency domain. 
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

freq_bins_pos_neg = np.fft.fftshift(np.fft.fftfreq(len(w_t), delta_t))
freq_bins_pos_neg[N//2] = freq_bins_pos_neg[N//2 + 1]

PSD_pos_neg = PowerSpectralDensity(freq_bins_pos_neg)


N = len(w_t)
delta_f = freq[2] - freq[1]
# - use positive and negative frequencies.
    
w_fft = np.fft.fftshift(np.fft.fft(w_t))  # Compute fft of windowing function (neg_pos_freq)
w_star_fft = np.conjugate(w_fft)  # Compute conjugate 

Cov_Matrix = np.zeros(shape=(N//2 + 1,N//2 + 1),dtype=complex) # Matrix will be filled full of complex numbers.
                                                               # here we only have positive frequencies

# Build analytical covariance matrix
print("Building the analytical covariance matrix")
Cov_Matrix_Gated = get_Cov(Cov_Matrix, w_fft, w_star_fft, delta_f, PSD_pos_neg)

os.chdir('Data/')
np.save("Cov_Matrix_analytical_gap.npy", Cov_Matrix_Gated)





