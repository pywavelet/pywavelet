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

freq, PSD = freq_PSD(t, delta_t)  # Extract frequency bins and PSD.


# Gaps in the frequency domain. 
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)

# ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================
print("Estimating the gated covariance matrix")
noise_f_gap_vec = []
for i in tqdm(range(0,100000)):
    np.random.seed(i)
    noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
    noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
    noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real

    noise_t_iter = np.fft.irfft(noise_f_iter)      # Compute stationary noise in TD
    noise_t_gap_iter = w_t * noise_t_iter  # Place gaps in the noise from the TD
    noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter) # Convert into FD
    noise_f_gap_vec.append(noise_f_gap_iter) 

# ==========================================================================================

print("Now estimating covariance matrix")    
cov_matrix_freq_gap = np.cov(noise_f_gap_vec,rowvar = False)
print("Finished estimating the covariance matrix")

os.chdir('Data/')
np.save("Cov_Matrix_estm_gap.npy", cov_matrix_freq_gap)
