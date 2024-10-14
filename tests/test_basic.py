from pywavelet.transforms.from_wavelets import from_wavelet_to_freq
from pywavelet.transforms.to_wavelets import from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, Wavelet
import numpy as np
import matplotlib.pyplot as plt

def test_pure_f0_transform(plot_dir):
    f0 = 1
    Nf = 8
    Nt = 4
    N = Nf*Nt
    dt = 0.1
    freq = np.fft.rfftfreq(N, dt)
    hf = np.zeros_like(freq, dtype=np.complex128)
    f0_idx = np.argmin(np.abs(freq - f0))
    hf[f0_idx] = 1.0
    freqseries = FrequencySeries(data=hf, freq=freq)
    wavelet = from_freq_to_wavelet(freqseries, Nf=Nf)
    freqseries_reconstructed = from_wavelet_to_freq(wavelet, dt=dt)

    plt.figure()
    plt.plot(
        freqseries.freq, np.abs(freqseries.data), 'o-', label=f"Original {freqseries.shape}"
    )
    plt.plot(
        freqseries_reconstructed.freq, np.abs(freqseries_reconstructed.data), '.', color='tab:red', label=f"Reconstructed {freqseries_reconstructed.shape}"
    )
    plt.legend()
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")

    plt.savefig(f"{plot_dir}/test_pure_f0_transform.png")

    assert np.allclose(freqseries.shape, freqseries_reconstructed.shape)

    # assert np.allclose(freqseries.data, freqseries_reconstructed.data)






