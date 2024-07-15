import numpy as np
import pytest

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.to_wavelets import from_time_to_wavelet
from pywavelet.transforms.types import TimeSeries
from pywavelet.utils.lisa import get_lisa_data
from pywavelet.utils.lvk import inject_signal_in_noise
from pywavelet.utils.snr import compute_snr

import matplotlib.pyplot as plt
from pastamarkers import markers


def test_toy_model_snr():

    A = 1e-3
    f0 = 1.0 * np.sqrt(3)
    T = 10000

    ########################################
    # Part1: Analytical SNR calculation
    ########################################
    dt = 0.2 / (
        2 * f0
    )  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array
    # round len(t) to the nearest power of 2
    t = t[: 2 ** int(np.log2(len(t)))]
    T = len(t) * dt

    N_t = len(t)
    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test
    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency
    y_fft = np.fft.fftshift(
        np.fft.fft(y)
    )  # continuous time fourier transform [seconds]

    import matplotlib.pyplot as plt
    # plt.stem(freq,abs(y_fft)**2,label = 'periodigram')
    # plt.xlabel(r'Frequency [Hz]')
    # plt.ylabel(r'Spectrogram')
    # plt.title('Checking Frequency Domain')
    # plt.show()


    print("Analytical result of periodigram is, not taking into account leakage ", abs(len(t))**2 * A**2 / 4)
    print("max value of periogram is = ",max(abs(y_fft)**2))

    # Now using sinusoids... this will be interesting


    m0 = f0 * N_t * dt


    analytical_fft = (A / 2*1j) * np.array([N_t if k == m0 or k == -m0 else 
        (np.sin((1) *(np.pi * (m0 - k))) / np.sin(np.pi / N_t * (m0 - k)) - 
        np.sin((1) * np.pi * (m0 + k)) / np.sin(np.pi / N_t * (m0 + k))) 
        for k in range(-N_t//2, N_t//2)
    ])

    print("Now with leakage formula, what is max value? ", max(abs(analytical_fft)**2))
    
    # analytical_fft = (A / 2*1j) * np.array([N_t if k == m0 or k == -m0 else 
    #     (np.exp(-1j * 2*np.pi/N_t * (m0 - k)* (N_t - 1)/2) * np.sin((np.pi * (m0 - k))) / np.sin(np.pi / N_t * (m0 - k)) - 
    #     np.exp(-1j * 2*np.pi/N_t * (m0 + k)* (N_t - 1)/2) * np.sin(np.pi * (m0 + k)) / np.sin(np.pi / N_t * (m0 + k))) 
    #     for k in range(-N_t//2, N_t//2)
    # ])

    # plt.plot(freq,abs(y_fft)**2,label = 'numerical', c = 'blue')
    # plt.plot(freq,abs(analytical_fft)**2, label = 'analytical', c = 'red')
    # plt.xlabel(r'Frequency [Hz]')
    # plt.ylabel(r'Spectrogram')
    # plt.title('Checking Frequency Domain')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(10, 6))  # Increase the figure size for better readability

    # Plotting the numerical data
    plt.plot(freq, abs(y_fft)**2, label='Numerical', color='blue', linestyle='-',  marker=markers.tortellini, alpha=0.7)

    # Plotting the analytical data
    plt.plot(freq, abs(analytical_fft)**2, label='Analytical', color='red', marker = markers.stelline, linestyle='--', alpha=0.5)

    # Adding labels and title
    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.ylabel('Spectrogram', fontsize=14)
    plt.title('Checking Frequency Domain', fontsize=16)

    # Adding legend
    plt.legend(fontsize=12)

    # Displaying the plot
    plt.grid(True)  # Adding grid for better readability
    plt.xlim([-f0-0.0002,-f0+0.0002])
    plt.show()
    breakpoint()