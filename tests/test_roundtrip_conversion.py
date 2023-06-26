import numpy as np
from scipy.signal.windows import tukey, hann
import matplotlib.pyplot as plt

from scipy.signal import chirp, spectrogram
import scipy

from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time
from pywavelet.transforms import from_freq_to_wavelet, from_wavelet_to_freq



def waveform_fft(t, waveform,):
    N = len(waveform)
    taper = tukey(N,0.1)
    waveform_w_pad = zero_pad(waveform*taper)
    waveform_f = np.fft.rfft(waveform_w_pad)[1:]

    n_t = len(zero_pad(t))
    delta_t = t[1]-t[0]
    freq = np.fft.rfftfreq(n_t,delta_t)[1:]
    return freq, waveform_f

def zero_pad(data):
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data,(0,int((2**pow_2)-N)),'constant')


# plot signal
def plot_signal(t, h):
    T = max(t)
    h_phase = np.arcsin(h)
    fs = 1/(t[1]-t[0])
    ff, tt, Sxx = spectrogram(h, fs=fs, nperseg=256, nfft=576)
    freq, h_freq = waveform_fft(t, h)

    fig, axes = plt.subplots(3,1, figsize=(4,6))
    axes[0].scatter(t, h, marker='.', c=h_phase, lw=0.1)
    axes[0].set_ylabel("h(t)")
    axes[0].set_xlim(0, T)
    axes[1].plot(freq, np.abs(h_freq))
    axes[1].set_ylabel("|h(f)|")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[2].set_xlim(0, T)
    axes[2].pcolormesh(tt, ff[:145], Sxx[:145],)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlim(0, T)
    plt.tight_layout()
    plt.show()

def generate_signal():
    fs = 4096
    T = 4
    t = np.arange(0, int(T * fs)) / fs
    h = chirp(t, f0=10, f1=1000, t1=T, method='quadratic')
    return t, h


def test_wavelet_to_freq_roundtrip():
    t, h = generate_signal()
    freq, h_freq = waveform_fft(t, h)
    plot_signal(t, h)

    Nt = 128
    Nf = 512
    h_wavelet = from_freq_to_wavelet(h_freq, Nf=Nf, Nt=Nt)
    h_time = from_wavelet_to_time(h_wavelet, Nf=Nf, Nt=Nt, mult=16)

    plot_signal(t, h_time)


def test_wavelet_to_time_roundtrip():
    t, h = generate_signal()

    plot_signal(t, h)

    Nt = 128
    Nf = 512
    h_wavelet = from_time_to_wavelet(h, Nf=Nf, Nt=Nt, mult=16)
    h_time = from_wavelet_to_time(h_wavelet, Nf=Nf, Nt=Nt, mult=16)

    plot_signal(t, h_time)