"""
Generate static assets used in the docs.

This script is not run during doc builds (notebooks execution is off). It is meant
to be run manually and the generated images committed into docs/_static.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.signal import chirp, spectrogram

from pywavelet.transforms import (
    from_time_to_wavelet,
    from_wavelet_to_time,
    omega,
    phi_vec,
    phitilde_vec_norm,
)
from pywavelet.types import TimeSeries

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"


def _save(fig: plt.Figure, name: str) -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    out = STATIC / name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


def make_phi_sweeps() -> None:
    Nf = 64
    Nt = 64
    mult = 16
    nxs = [1.0, 4.0, 16.0]

    # phi(t)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    for nx in nxs:
        phi = phi_vec(Nf, d=nx, q=mult)
        phi = phi / np.max(np.abs(phi))
        x = np.arange(len(phi)) - (len(phi) // 2)
        ax.plot(x, phi, label=f"nx={nx:g}")
    ax.set_title("Time-domain window $\\phi$ (normalized)")
    ax.set_xlabel("sample index (centered)")
    ax.set_ylabel("amplitude")
    ax.legend(frameon=False, ncol=len(nxs))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "phi_sweep.png")

    # phitilde(ω)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    w = omega(Nf, Nt)
    for nx in nxs:
        phit = phitilde_vec_norm(Nf, Nt, d=nx)
        phit = phit / np.max(np.abs(phit))
        ax.plot(w, phit, label=f"nx={nx:g}")
    ax.set_title("Frequency-domain window $\\~\\phi(\\omega)$ (normalized)")
    ax.set_xlabel("angular frequency $\\omega$")
    ax.set_ylabel("amplitude")
    ax.legend(frameon=False, ncol=len(nxs))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "phitilde_sweep.png")


def make_quick_visualization() -> None:
    Nf, Nt = 64, 64
    nx = 4.0
    mult = 16

    phi = phi_vec(Nf, d=nx, q=mult)
    w = omega(Nf, Nt)
    phitilde = phitilde_vec_norm(Nf, Nt, d=nx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(phi)
    axes[0].set_title(r"$\phi$ (time domain)")
    axes[0].set_xlabel("sample index")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(w, phitilde)
    axes[1].set_title(r"$\~\phi(\omega)$ (frequency domain)")
    axes[1].set_xlabel(r"angular frequency $\omega$")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, "window_quick_visualization.png")


def make_window_effects_wavelet() -> None:
    dt = 1 / 512
    Nf = 64
    Nt = 64
    ND = Nf * Nt

    t = np.arange(ND) * dt
    y = chirp(t, f0=10.0, f1=120.0, t1=t[-1], method="hyperbolic")
    ts = TimeSeries(data=y, time=t)

    # Original signal overview
    fs = 1.0 / dt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(t, y, color="#0969da", linewidth=1)
    axes[0].set_title("Original signal (time domain)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.25)

    f, tt, Sxx = spectrogram(y, fs=fs, nperseg=256, noverlap=192)
    im = axes[1].pcolormesh(
        tt, f, Sxx, shading="nearest", cmap="Reds", norm=LogNorm(vmin=1e-12)
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("frequency [Hz]")
    axes[1].set_ylim(0, fs / 2)
    cbar = fig.colorbar(im, ax=axes[1], pad=0.02)
    cbar.set_label("power")
    fig.tight_layout()
    _save(fig, "window_effects_original_signal.png")

    configs = [
        {"nx": 1.0, "mult": 16},
        {"nx": 4.0, "mult": 16},
        {"nx": 8.0, "mult": 16},
        {"nx": 4.0, "mult": 8},
    ]

    waves = []
    stats = []
    residuals = []
    for cfg in configs:
        wave = from_time_to_wavelet(
            ts, Nf=Nf, Nt=Nt, nx=cfg["nx"], mult=cfg["mult"]
        )
        recon = from_wavelet_to_time(
            wave, dt=dt, nx=cfg["nx"], mult=cfg["mult"]
        )
        residual = ts.data - recon.data
        rms = float(np.sqrt(np.mean(residual**2)))
        maxabs = float(np.max(np.abs(residual)))
        waves.append(wave)
        stats.append({"rms": rms, "maxabs": maxabs})
        residuals.append(np.array(residual))

    # Shared color scale (log) across panels.
    z_all = np.concatenate([np.abs(w.data).ravel() for w in waves])
    z_pos = z_all[z_all > 0]
    vmin = float(np.percentile(z_pos, 5)) if z_pos.size else 1e-12
    vmax = float(np.percentile(z_pos, 99.5)) if z_pos.size else 1.0
    vmin = max(vmin, 1e-12)
    vmax = max(vmax, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    im = None
    for ax, cfg, wave, st in zip(axes, configs, waves, stats):
        z = np.abs(np.array(wave.data))
        im = ax.imshow(
            z,
            aspect="auto",
            origin="lower",
            norm=norm,
            extent=[wave.time[0], wave.time[-1], wave.freq[0], wave.freq[-1]],
            interpolation="nearest",
            cmap="Reds",
        )
        ax.set_title(
            f"nx={cfg['nx']:g}, mult={cfg['mult']}\n"
            f"resid rms={st['rms']:.2e}, max={st['maxabs']:.2e}",
            fontsize=10,
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")

    assert im is not None
    fig.suptitle(
        "Effect of window parameters on wavelet coefficients and reconstruction",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.94])

    cax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$|W|$ (log scale)")
    _save(fig, "window_effects_wavelet.png")

    # Residual time series panels (how the signal changes)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
    axes = axes.ravel()
    lim = (
        float(np.max([np.max(np.abs(r)) for r in residuals]))
        if residuals
        else 1.0
    )
    for ax, cfg, r, st in zip(axes, configs, residuals, stats):
        ax.plot(t, r, color="#cf222e", linewidth=1)
        ax.axhline(0.0, color="#6e7781", linewidth=1, alpha=0.6)
        ax.set_title(
            f"nx={cfg['nx']:g}, mult={cfg['mult']}  (rms={st['rms']:.2e})",
            fontsize=10,
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("residual")
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "window_effects_residuals.png")


def main() -> None:
    make_phi_sweeps()
    make_quick_visualization()
    make_window_effects_wavelet()


if __name__ == "__main__":
    main()
