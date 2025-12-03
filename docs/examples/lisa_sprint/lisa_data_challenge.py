import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from log_psplines.psplines import LogPSplines
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey
from sympy.physics.units import percent

from pywavelet import set_backend

set_backend("jax")

import jax.numpy as jnp

from pywavelet.transforms import from_freq_to_wavelet, from_wavelet_to_freq
from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

plot_dir = "out_spritz"
os.makedirs(plot_dir, exist_ok=True)


def plot_wnm_and_ts(timeseries, freqseries, wnm, label="spritz"):
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    timeseries.plot(ax=ax[0])
    freqseries.plot_periodogram(ax=ax[1])
    wnm.plot(zscale="log", absolute=True, ax=ax[2])
    fig.suptitle(label)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{label}.png"))


def plot_mask(wnm, mask, label="mask"):
    mask_wnm = deepcopy(wnm)
    mask_wnm.data = mask_wnm.data.at[mask].set(1)  # set outliers to 1 (masked)
    mask_wnm.data = mask_wnm.data.at[~mask].set(
        0
    )  # set non-outliers to 0 (unmasked -- we keep)
    d = wnm.data.ravel()
    percentage = np.sum(mask) / len(d)
    print(f"Percentage of data masked: {percentage:.2%}")

    mask_wnm.plot(
        absolute=True,
        cmap="binary",
        show_colorbar=False,
        label=f"{percentage:.2%}% masked",
        zscale="linear",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{label}.png"))


def mad_threshold(wnm: Wavelet, nsigma: float = 5.0) -> Wavelet:
    """
    Sigma-clipping using median and MAD across time for each frequency bin.
    """
    print(f"MAD thresholding -- deleting values > nsigma ({nsigma:.2f})* MAD")

    wnm_copy = deepcopy(wnm)
    data = jnp.abs(wnm.data)
    # median and MAD
    median = jnp.median(data)
    mad = jnp.median(np.abs(data - median))

    # mask outliers (True = outlier)
    mask = np.abs(data - median) > nsigma * mad
    wnm_copy.data = wnm_copy.data.at[mask].set(0)
    return wnm_copy, mask


def iterative_denoise(
    wnm: Wavelet,
    iterations: int = 5,
    base_threshold: float = 0.3,
    decay: float = 0.9,
    **kwargs,
) -> Wavelet:
    """
    Iteratively apply thresholding, decaying the threshold each time.
    method: 'global', 'mad', or 'blockwise'
    """
    new_wnm = deepcopy(wnm)
    for i in range(iterations):
        thr = base_threshold * (decay**i)
        new_wnm, mask = mad_threshold(new_wnm, nsigma=thr * 10)
        plot_mask(wnm, mask, label=f"mask-{i}")

        # plot each iteration
        fs = new_wnm.to_frequencyseries()
        ts = fs.to_timeseries()
        plot_wnm_and_ts(ts, fs, new_wnm, label=f"Iter{i + 1}--thr{thr:.3f}")

    return new_wnm


f = "/Users/avaj0001/Downloads/drive-download-20250430T050734Z-1-001/LDC-2b-Spritz-MBHB1.asc"
data = np.loadtxt(f, skiprows=2).T
t, X, Y, Z = data

ts = TimeSeries(X, time=t)
fs = ts.to_frequencyseries()

orig_wnm = fs.to_wavelet(Nf=256)
plot_wnm_and_ts(ts, fs, orig_wnm, label="Origial Spritz")

wnm = fs.to_wavelet(Nf=256)

wnm = iterative_denoise(wnm, iterations=5, base_threshold=0.9, decay=0.9)
