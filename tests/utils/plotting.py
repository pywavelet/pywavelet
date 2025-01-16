import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

__all__ = [
    "plot_wavelet_comparison",
    "plot_timedomain_comparisons",
    "plot_freqdomain_comparisions",
    "BRANCH",
    "plot_fft",
]


def __get_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else "main"
    return branch


BRANCH = __get_branch()


def __symlog_bins(
    data: np.ndarray, lin_thresh=None, log_base=10, num_lin=5, num_log=10
):
    if lin_thresh is None:
        lin_thresh = 0.05 * np.std(data)
    lin_bins = np.linspace(-lin_thresh, lin_thresh, num_lin)
    log_bins_pos = np.logspace(
        np.log10(lin_thresh),
        np.log10(np.max(np.abs(data))),
        num=num_log,
        base=log_base,
    )
    log_bins_neg = -np.logspace(
        np.log10(lin_thresh),
        np.log10(np.max(np.abs(data))),
        num=num_log,
        base=log_base,
    )[::-1]
    bins = np.concatenate([log_bins_neg, lin_bins, log_bins_pos])
    bins = np.sort(np.unique(bins))
    return bins, lin_thresh


def plot_residuals(residuals: np.ndarray, ax: plt.Axes, symlog=True):
    mean, std = np.mean(residuals), np.std(residuals)
    ax.text(
        0.05,
        0.95,
        r"${:.2f} \pm {:.2f}$".format(mean, std),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    num_nans = np.sum(np.isnan(residuals))
    assert num_nans == 0, f"Found {num_nans} NaNs in residuals."
    if symlog:
        bins, lin_thresh = __symlog_bins(residuals)
        ax.hist(residuals, bins=bins, density=False)
        ax.set_xscale("symlog", linthresh=lin_thresh)
    else:
        ax.hist(residuals, bins=100, density=False)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")


def plot_wavelet_comparison(
    cur: Wavelet, cached: Wavelet, err: Wavelet, label: str, outdir: str
):
    net_err = np.sum(np.abs(err.data))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
    axes[0].set_title(f"Branch: {BRANCH}")
    axes[1].set_title("Cached (v0.0.1)")
    axes[2].set_title("Diff")
    textstr = f"Σ|old-new|: {net_err:.2e}"
    props = dict(boxstyle="round", facecolor=None, alpha=0.2)
    axes[2].text(
        0.05,
        0.85,
        textstr,
        transform=axes[2].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    vmin = min([np.min(cur.data), np.min(cached.data), -1])
    vmax = max([np.max(cur.data), np.max(cached.data), 1])
    norm = TwoSlopeNorm(0, vmin, vmax)
    cur.plot(ax=axes[0], norm=norm, cmap="bwr", show_colorbar=False)
    cached.plot(ax=axes[1], norm=norm, cmap="bwr", show_colorbar=True)
    err.plot(ax=axes[2], cmap="bwr", show_colorbar=True, cbar_label="old-new")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("Time [s]")
    plt.savefig(f"{outdir}/{label}_comparison.png")
    return cached


def plot_timedomain_comparisons(
    ht: TimeSeries, h_reconstructed: TimeSeries, wavelet: Wavelet, fname: str
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ht.plot(ax=axes[0], label="Original")
    h_reconstructed.plot(
        ax=axes[0],
        label="Reconstructed",
        linestyle="--",
        color="tab:orange",
        alpha=0.5,
    )
    axes[0].legend()
    wavelet.plot(ax=axes[1])
    r = ht.data - h_reconstructed.data
    plot_residuals(ht.data - h_reconstructed.data, axes[2])
    axes[0].set_title("Timeseries")
    axes[1].set_title("Wavelet")
    axes[2].set_title("Residuals")
    plt.tight_layout()
    plt.savefig(fname)


def plot_freqdomain_comparisions(
    hf: FrequencySeries,
    h_reconstructed: FrequencySeries,
    wavelet: Wavelet,
    fname: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hf.plot_periodogram(ax=axes[0], label="Original")
    h_reconstructed.plot_periodogram(
        ax=axes[0],
        label="Reconstructed",
        linestyle="--",
        color="tab:orange",
        alpha=0.5,
    )
    axes[0].legend()
    wavelet.plot(ax=axes[1])
    try:
        r = np.abs(hf.data) - np.abs(h_reconstructed.data)
        plot_residuals(r, axes[2])
    except Exception as e:
        print(e)

    axes[2].set_title("Residuals (in WDF f-range)")
    axes[0].set_title("Periodogram")
    axes[1].set_title("Wavelet")
    plt.tight_layout()
    plt.savefig(fname)


def plot_fft(hf, hf_1, fname):
    plt.figure()
    plt.plot(hf.freq, np.abs(hf.data), "o-", label=f"Original {hf.shape}")
    plt.plot(
        hf_1.freq,
        np.abs(hf_1.data),
        ".",
        color="tab:red",
        label=f"Reconstructed {hf_1.shape}",
    )
    plt.legend()
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.savefig(fname)
