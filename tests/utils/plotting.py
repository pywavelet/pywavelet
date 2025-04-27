import subprocess
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants.codata2018 import alpha
from matplotlib.colors import LogNorm
from numpy.ma.core import absolute

from pywavelet.types import FrequencySeries, TimeSeries, Wavelet

__all__ = [
    "plot_wavelet_cached_comparison",
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


def plot_residuals(
    residuals: np.ndarray, ax: plt.Axes, symlog=False, log_bins=True
):
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
    if log_bins:
        bins = np.logspace(
            np.log10(np.min(np.abs(residuals))),
            np.log10(np.max(np.abs(residuals))),
            100,
        )
        ax.hist(residuals, bins=bins, density=False)
        ax.set_xscale("log")
    else:
        ax.hist(residuals, bins=100, density=False)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")


def plot_wavelet_comparison(
    w1: Wavelet,
    w2: Wavelet,
    labels=None,
    axes: plt.Axes = None,
    fname=None,
):
    if labels is None:
        labels = ["W1", "W2"]

    err = w1 - w2
    net_err = np.sum(np.abs(err.data))
    errstr = f"Σ|{labels[0]}-new|: {net_err:.2e}"

    if axes is None:
        fig, axes = plt.subplots(
            1, 3, figsize=(10, 4), sharey=True, sharex=True
        )

    txtbox_kwargs = dict(alpha=0.5, facecolor="white")
    norm = _get_log_norm(w1.data, w2.data, default_vmin=1e-10)
    kwgs = dict(
        norm=norm, zscale="log", absolute=True, txtbox_kwargs=txtbox_kwargs
    )
    w1.plot(ax=axes[0], show_colorbar=False, **kwgs)
    w2.plot(ax=axes[1], show_colorbar=False, **kwgs)
    err.plot(
        ax=axes[2],
        show_colorbar=True,
        cbar_label=f"{labels[0]}-{labels[1]}",
        label=errstr,
        **kwgs,
    )

    for i, ax in enumerate(axes):
        ax.set_ylabel("Frequency [Hz]" if i == 0 else "")
        ax.set_xlabel("Time [s]")
    if fname:
        plt.savefig(fname)
    return axes


def plot_wavelet_cached_comparison(
    cur: Wavelet, cached: Wavelet, err: Wavelet, label: str, outdir: str
):
    from pywavelet.backend import current_backend

    axes = plot_wavelet_comparison(cur, cached, labels=["Current", "Cached"])
    axes[0].set_title(f"Branch: {BRANCH} [{current_backend}]")
    axes[1].set_title("Cached Numpy (v0.0.1)")
    axes[2].set_title("Diff")
    plt.suptitle(label)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_comparison.png")
    return cached


def _get_log_norm(cur_data, cached_data, default_vmin=1e-10):
    abs_cur_data = np.abs(cur_data)
    abs_cached_data = np.abs(cached_data)

    vmin_cur = (
        np.min(abs_cur_data[abs_cur_data > 0])
        if len(abs_cur_data[abs_cur_data > 0]) > 0
        else default_vmin
    )
    vmin_cached = (
        np.min(abs_cached_data[abs_cached_data > 0])
        if len(abs_cached_data[abs_cached_data > 0]) > 0
        else default_vmin
    )

    vmin = min(vmin_cur, vmin_cached)
    vmax = max(np.max(abs_cur_data), np.max(abs_cached_data))

    return LogNorm(vmin, vmax)


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
    error = np.sqrt((ht.data - h_reconstructed.data) ** 2)
    plot_residuals(error, axes[2])
    axes[0].set_title("Timeseries")
    axes[1].set_title("Wavelet")
    axes[2].set_title("Residuals")
    plt.tight_layout()
    plt.savefig(fname)


def plot_freqdomain_comparisions(
    hf: FrequencySeries,
    h_reconstructed: FrequencySeries,
    wavelet: Wavelet,
    fname: str = None,
    axes=None,
):
    err = np.abs(hf.data - h_reconstructed.data)
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hf.plot_periodogram(ax=axes[0], label=f"Original (n={hf.ND})")
    h_reconstructed.plot_periodogram(
        ax=axes[0],
        label=f"Reconstructed (n={h_reconstructed.ND})",
        linestyle="--",
        color="k",
        alpha=0.9,
    )
    axes[0].set_xscale("linear")
    ax_diff = axes[0].twinx()
    ax_diff.plot(
        hf.freq,
        err,
        color="tab:red",
        label="Diff",
        alpha=0.2,
        zorder=-1,
    )

    # add diff to legend
    axes[0].legend(loc="upper left", frameon=False)
    ax_diff.legend(loc="upper right", frameon=False)
    # put ticks + ticklabels INSIDE (and in red
    ax_diff.tick_params(
        axis="y",
        colors="red",
        direction="in",
        labelleft=True,
        labelright=False,
    )
    ax_diff.yaxis.label.set_color("red")
    ax_diff.spines["right"].set_color("red")
    ax_diff.set_yscale("log")

    wavelet.plot(ax=axes[1])
    try:
        plot_residuals(err, axes[2], log_bins=True)
    except Exception as e:
        print(e)

    axes[2].set_title("Residuals (freq-domain)")
    axes[0].set_title("Periodogram")
    axes[1].set_title("Wavelet")

    xpad = 5
    axes[0].set_xscale("linear")
    axes[0].set_xlim(-xpad, hf.freq[-1] + xpad)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        return axes


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


def plot_roundtrip(
    orig_freq: FrequencySeries,
    orig_wdm: Wavelet,
    reconstructed_freq: FrequencySeries,
    reconstructed_wdm: Wavelet,
    axes=None,
    labels=None,
):
    """Plots the
    Axes1: original frequency series + reconstructed frequency series
    Axes2: residual of the frequency series
    Axes3: original wavelet
    Axes4: reconstructed wavelet
    Axes5: diff of the wavelet
    """
    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(10, 8))

    if labels is None:
        labels = ["Original", "Reconstructed"]

    # relative error in the frequency domain

    err_freq = (
        np.abs(
            np.abs(orig_freq.data) ** 2 - np.abs(reconstructed_freq.data) ** 2
        )
        / np.abs(orig_freq.data) ** 2
    )

    err_wdm = orig_wdm - reconstructed_wdm
    net_err_freq = np.sum(np.abs(err_freq))
    net_err_wdm = np.sum(np.abs(err_wdm.data))

    ax_err = axes[0].twinx()
    ax_err.plot(
        orig_freq.freq,
        err_freq,
        color="tab:red",
        label=f"Err {net_err_freq:.2f}",
        lw=2,
        zorder=-1,
        alpha=0.5,
    )
    ax_err.set_yscale("log")
    ax_err.set_ylabel("Rel Error")
    # color ticks and axes red
    ax_err.tick_params(axis="y", colors="red")
    ax_err.yaxis.label.set_color("red")
    ax_err.spines["right"].set_color("red")

    orig_freq.plot_periodogram(
        ax=axes[0], label=labels[0], marker="o", color="black"
    )
    reconstructed_freq.plot_periodogram(
        ax=axes[0],
        label=f"{labels[1]} (er={net_err_freq:.2f})",
        linestyle="--",
        marker="o",
        color="tab:blue",
        alpha=0.5,
    )
    # extend the xlim a bit to the right
    axes[0].set_xlim(
        orig_freq.freq[0] - 5 * orig_freq.df,
        orig_freq.freq[-1] + 5 * orig_freq.df,
    )
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title(f"Frequency Series")
    orig_wdm.plot(ax=axes[1], label=labels[0], absolute=True, cmap="Blues")
    reconstructed_wdm.plot(
        ax=axes[2], label=labels[1], absolute=True, cmap="Blues"
    )
    err_wdm.plot(
        ax=axes[3],
        label=f"Diff, er={net_err_wdm:.2f}",
        absolute=True,
        cmap="Blues",
    )
