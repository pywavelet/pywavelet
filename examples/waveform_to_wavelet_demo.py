import numpy as np
import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt




def plot_spectrogram(title, t, fs, amp):
    ff, tt, Sxx = spectrogram(amp, fs=fs)
    fig, axes = plt.subplots(2,1, sharex=True)
    fig.suptitle(title)
    axes[0].plot(t, amp)
    axes[0].set_ylabel('Amplitude')
    axes[1].pcolormesh(tt, ff, Sxx, cmap='hot', shading='gouraud')
    axes[1].set_xlabel('t (sec)')
    axes[1].set_ylabel('Frequency (Hz)')
    plt.show()


fs = 1000
T = 2
t = np.arange(0, int(T*fs)) / fs
title = "Linear Chirp, f(0)=6, f(10)=1"
amp = chirp(t, f0=1, f1=50, t1=T, method='quadratic')
plot_spectrogram(title, t, fs, amp)
