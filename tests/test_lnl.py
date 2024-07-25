import numpy as np
import pytest
from numpy.random import normal

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries
from pywavelet.utils.lisa import FFT, get_lisa_data, waveform, zero_pad
from pywavelet.utils.lvk import inject_signal_in_noise
from pywavelet.utils.snr import compute_snr


def test_lisa_lnl(plot_dir):
    np.random.seed(1234)
    a_true = 5e-21
    f_true = 1e-3
    fdot_true = 1e-8
    h_t, h_f, psd, SNR = get_lisa_data(a_true, f_true, fdot_true, 0.033)
    t = h_t.time
    n = len(h_t)
    nf = len(psd)
    variance_noise_f = n * psd.data / (4 * h_t.dt)
    var = np.sqrt(variance_noise_f)
    noise_f = normal(0, var, nf) + 1j * normal(0, var, nf)
    template_f = h_f.data
    data_f = template_f + 0 * noise_f  # Construct data stream
    data_f = FrequencySeries(data=data_f, freq=h_f.freq)
    kwgs = dict(
        Nf=256,
        mult=16,
    )
    d = Data.from_frequencyseries(data_f, **kwgs).wavelet
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=d.freq,
        t_grid=d.time,
        dt=h_t.dt,
    )

    FM_estimate = a_true / SNR
    a_vals = np.linspace(
        a_true - 5 * FM_estimate, a_true + 5 * FM_estimate, 100
    )

    n = len(zero_pad(waveform(a_true, f_true, fdot_true, t)))
    variance_noise_f = n * psd.data / (4 * h_t.dt)

    def lnl_func(a):
        ht = waveform(a, f_true, fdot_true, t)
        hf = FFT(ht, taper=False)
        return -0.5 * sum((abs(data_f - hf) ** 2) / variance_noise_f)

    def wavelet_lnl_func(a):
        ht = waveform(a, f_true, fdot_true, t)
        hf = FrequencySeries(FFT(ht), h_f.freq)
        h = Data.from_frequencyseries(hf, **kwgs).wavelet
        return -0.5 * np.nansum(((d - h) ** 2) / psd_wavelet)

    # Compute SNR, wavelets
    ht = waveform(a_true, f_true, fdot_true, t)
    hf_discrete = FrequencySeries(FFT(ht), h_f.freq)
    hf_continuous = FrequencySeries(FFT(ht), h_f.freq)
    h = Data.from_frequencyseries(hf_continuous, **kwgs).wavelet

    SNR2_wavelet = np.nansum((h * h) / psd_wavelet)
    SNR2_freq = (
        4 * h_t.dt * np.sum(abs(hf_discrete.data) ** 2 / (n * psd.data))
    )

    print("SNR wavelet  = ", SNR2_wavelet ** (1 / 2))
    print("SNR freq = ", SNR2_freq ** (1 / 2))

    lnl = [lnl_func(a) for a in a_vals]
    lnl_wavelets = [wavelet_lnl_func(a) for a in a_vals]
    # lnl_wavelets = np.zeros(len(a_vals))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(a_vals, lnl)
    axes[1].plot(a_vals, lnl_wavelets)
    axes[0].set_ylabel("LnL (freq domain)")
    axes[1].set_ylabel("LnL (WDM domain)")
    for ax in axes:
        ax.axvline(a_true, color="k", linestyle="--")
        ax.set_xlabel("A from $A\sin(2\pi (ft + 0.5 * \dot{f} t^2))$")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/lnl.png", dpi=300)

    # plot likelihood with FM estimate.
    A = a_true
    plt.figure()
    plt.plot(a_vals, np.exp(np.array(lnl_wavelets)), label="WDM domain")
    plt.axvline(x=A - FM_estimate, label="FM", c="black")
    plt.axvline(x=A + FM_estimate, label="FM", c="black")
    # twin axes
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(
        a_vals,
        np.exp(np.array(lnl)),
        ls="--",
        color="tab:orange",
        label="Freq domain",
    )
    ax2.legend(loc="upper right", frameon=False)
    ax.set_ylabel("Posterior")
    ax.legend(loc="upper left", frameon=False)
    plt.savefig(f"{plot_dir}/lnl_fm.png", dpi=300)


def test_lvk_lnl(plot_dir):
    # FFT is different, need to divide by delta_t
    Nf = 128
    signal_f, psd, snr = inject_signal_in_noise(mc=30, noise=False)
    
    data = Data.from_frequencyseries(
        signal_f,
        Nf=Nf,
        mult=32,
    )
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=data.wavelet.freq.data,
        t_grid=data.wavelet.time.data,
        dt=signal_t.dt,
    )
    data.plot_wavelet()
    wavelet_snr = compute_snr(data.wavelet, psd_wavelet)
    assert np.isclose(
        snr, wavelet_snr, atol=1
    ), f"LVK SNR mismatch {snr} != {wavelet_snr}"
