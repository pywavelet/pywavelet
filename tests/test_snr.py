import numpy as np
import pytest

from pywavelet.transforms.to_wavelets import from_time_to_wavelet, from_freq_to_wavelet
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd
import matplotlib.pyplot as plt


# pytest parameterize decorator
@pytest.mark.parametrize(
    "log2_T, A, PSD_AMP, Nf",
    [
        (14, 1e-2, 1, 64),
        # (0.1, 12, 1e-1, 1e-2, 32),
        # (0.1, 12, 1e-1, 1e-2, 64),
    ],
)
def test_toy_model_snr(log2_T, A, PSD_AMP, Nf):
    ########################################
    # Part1: Analytical SNR calculation
    #######################################
    dt = 1
    T = 2 ** log2_T  # Total time
    t = np.arange(0, T, dt)  # Time array
    ND = len(t)  # Number of data points
    f0 = 1 / 16
    # assert f0 * ND is an integer
    assert np.isclose(f0 * ND, int(f0 * ND), atol=1e-5), f"f0={f0} * ND={ND} must be an integer"


    # round len(t) to the nearest power of 2
    assert f0 <= 0.5 / dt, f"f0={f0} must be less than Nyquist frequency={0.5/dt}"


    # Eq 21
    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test

    # makes the freq -> [-0.5,...0,... 0.5] Hz
    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency
    y_fft = dt * np.fft.fftshift(
        np.fft.fft(y)
    )  # continuous time fourier transform [seconds]

    PSD = PSD_AMP * np.ones(len(freq))  # PSD of the noise

    # Compute the SNRs
    SNR2_f = 2 * np.sum(abs(y_fft) ** 2 / PSD) * df
    SNR2_t = 2 * dt * np.sum(abs(y) ** 2 / PSD)
    SNR2_t_analytical = (A**2) * T / PSD[0]

    # assert np.isclose(
    #     SNR2_t, SNR2_t_analytical, atol=0.5
    # ), f"{SNR2_t}!={SNR2_t_analytical}"
    # assert np.isclose(
    #     SNR2_f, SNR2_t_analytical, atol=0.5
    # ), f"{SNR2_f}!={SNR2_t_analytical}"

    ########################################
    # Part2: Wavelet domain
    ########################################
    ND = len(y)
    Nt = ND // Nf
    assert Nt > 1 , f"Nt={Nt} must be greater than 1 (ND={ND}, Nf={Nf})"
    signal_timeseries = TimeSeries(y, t)
    signal_freq = signal_timeseries.to_frequencyseries()
    signal_wavelet = from_time_to_wavelet(signal_timeseries, Nf=Nf, Nt=Nt)

    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet.freq,
        t_grid=signal_wavelet.time,
        dt=dt,
    )
    wavelet_snr2 = compute_snr(signal_wavelet, psd_wavelet) ** 2



    #### PLOTTING

    fig, ax = signal_wavelet.plot()
    fig.suptitle(f"SNR2_t{SNR2_t:.2f}, SNR2_wdm={wavelet_snr2:.2f}")
    fig.savefig(f"TEST_NfNf{Nf}x{Nt}_snr{SNR2_f}.png")


    assert np.isclose(
        wavelet_snr2, SNR2_t, atol=1e-2
    ), f"SNRs dont match {wavelet_snr2:.2f}!={SNR2_t:.2f} (factor:{SNR2_t/wavelet_snr2:.2f})"



    Nt = ND // Nf
    signal_wavelet_f = from_freq_to_wavelet(signal_freq, Nf=Nf, Nt=Nt)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
            psd=PSD,
            psd_f=freq,
            f_grid=signal_wavelet_f.freq,
            t_grid=signal_wavelet_f.time,
            dt=dt,
        )
    wavelet_snr2 = compute_snr(signal_wavelet_f, psd_wavelet) ** 2



    #### PLOTTING

    fig, ax = signal_wavelet_f.plot()
    fig.suptitle(f"SNR2_t{SNR2_t:.2f}, SNR2_wdm={wavelet_snr2:.2f}")
    fig.savefig(f"TEST_NfNf{Nf}x{Nt}_snr{SNR2_f}_freq.png")


    assert np.isclose(
        wavelet_snr2, SNR2_t, atol=1e-2
    ), f"SNRs dont match {wavelet_snr2:.2f}!={SNR2_t:.2f} (factor:{SNR2_t/wavelet_snr2:.2f})"


    # Analytical wavelet SNR

    #  A22Nf [Nt/2]
    # Snm

    """
    given f0 = m0/(N ∆t)
    m0 = f0 N ∆t
    
    ωnm =
        (
        
            A sqrt(2 * Nf) if time_bin is +ive and odd
                            and freq_bin is 2*m0/Nf
            else
            0       
    """

    analytical_wnm =np.zeros((Nt, Nf))
    m0 = int(f0 * Nf * dt)
    mask = np.zeros((Nt, Nf))

    # all odd +ive time bins
    mask[1::2, 2*m0/Nt] = 1
    analytical_wnm = A * np.sqrt(2 * Nf) * mask
    analytical_wnm = Wavelet(analytical_wnm.T, signal_wavelet.time, signal_wavelet.freq)

    # analytical_wnr_snr2=





    # signal_wavelet.plot(ax=axes[0], absolute=True)
    # analytical_wnm.plot(ax=axes[1], absolute=True)

    # axtwi = axes[1].twiny()
    # axtwi.set_yticks(np.arange(Nf))


    # imshow both , extentss of (0, Nt), (0, Nf)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].set_title("Time->Wavelet")
    im= axes[0].imshow(np.abs(signal_wavelet.data), aspect='auto', origin='lower', extent=(0, Nt, 0, Nf))
    fig.colorbar(im, ax=axes[0])

    axes[1].set_title("Freq->Wavelet")
    im =axes[1].imshow(np.abs(signal_wavelet_f.data), aspect='auto', origin='lower', extent=(0, Nt, 0, Nf))
    fig.colorbar(im, ax=axes[1])

    axes[2].set_title("Analytical Wavelet")
    im = axes[2].imshow(np.abs(analytical_wnm.data), aspect='auto', origin='lower', extent=(0, Nt, 0, Nf))
    fig.colorbar(im, ax=axes[2])

    fig.savefig("true_wdm.png")



