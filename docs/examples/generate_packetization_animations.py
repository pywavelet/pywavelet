"""
Generate nicer packetization animations for the docs.

This script is intentionally "offline": Jupyter Book execution is disabled, so we
generate assets once and commit them into docs/_static.

Outputs (written into docs/_static):
- time_to_wavelet.gif
- freq_to_wavelet.gif
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LogNorm
from scipy.signal import chirp

# Ensure deterministic, NumPy-based outputs regardless of the user's environment.
# (Backend selection happens at import time.)
os.environ.setdefault("PYWAVELET_BACKEND", "numpy")
os.environ.setdefault("PYWAVELET_PRECISION", "float64")

from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
from pywavelet.transforms import phi_vec, phitilde_vec_norm
from pywavelet.types import FrequencySeries, TimeSeries


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"


def _ensure_outdir() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)


def _save_gif(anim: animation.FuncAnimation, name: str, fps: int = 8) -> None:
    _ensure_outdir()
    out = STATIC / name
    writer = animation.PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=110)
    print(f"Wrote {out}")


def _chirp_timeseries(dt: float, Nf: int, Nt: int) -> TimeSeries:
    ND = Nf * Nt
    t = np.arange(ND) * dt
    y = chirp(t, f0=10.0, f1=120.0, t1=t[-1], method="hyperbolic")
    return TimeSeries(data=y, time=t)


# ----------------------------
# Frequency -> wavelet animation
# ----------------------------


def _dx_ifft_for_fbin(
    data: np.ndarray, phif: np.ndarray, Nf: int, Nt: int, f_bin: int
) -> np.ndarray:
    """
    Pure-Python version of the core idea in
    src/pywavelet/transforms/numpy/forward/from_freq.py.

    Builds DX for a given f_bin and returns DX_trans = ifft(DX).
    """
    DX = np.zeros(Nt, dtype=np.complex128)
    i_base = Nt // 2
    jj_base = f_bin * (Nt // 2)

    # center
    if f_bin == 0 or f_bin == Nf:
        DX[i_base] = phif[0] * data[jj_base] / 2.0
    else:
        DX[i_base] = phif[0] * data[jj_base]

    start = jj_base + 1 - (Nt // 2)
    end = jj_base + (Nt // 2)
    for jj in range(start, end):
        j = abs(jj - jj_base)
        i = i_base - jj_base + jj
        if (f_bin == Nf and jj > jj_base) or (f_bin == 0 and jj < jj_base):
            DX[i] = 0.0
        elif j == 0:
            continue
        else:
            DX[i] = phif[j] * data[jj]

    return np.fft.ifft(DX, Nt)


def _pack_wave_column_from_dx(dx_trans: np.ndarray, Nt: int, f_bin: int) -> np.ndarray:
    """Mimic the parity packing logic into wave[:, f_bin] for one column."""
    col = np.zeros(Nt, dtype=float)
    if f_bin == 0:
        col[0:Nt:2] = dx_trans[0:Nt:2].real * np.sqrt(2.0)
        return col
    if f_bin < 0:
        return col
    # f_bin == Nf handled outside (stored into column 0 odd rows)
    for n in range(Nt):
        if f_bin % 2:
            col[n] = -dx_trans[n].imag if (n + f_bin) % 2 else dx_trans[n].real
        else:
            col[n] = dx_trans[n].imag if (n + f_bin) % 2 else dx_trans[n].real
    return col


def make_freq_to_wavelet_gif() -> None:
    Nf = 32
    Nt = 32
    dt = 1 / 256
    nx = 4.0

    ts = _chirp_timeseries(dt, Nf=Nf, Nt=Nt)
    yf = np.fft.rfft(ts.data)
    power = np.abs(yf) ** 2

    phif = phitilde_vec_norm(Nf, Nt, d=nx)
    phif = np.asarray(phif, dtype=np.float64)

    # Precompute per-frame packets
    dx_packets = [(_dx_ifft_for_fbin(yf, phif, Nf, Nt, fb)) for fb in range(Nf + 1)]

    # Build wavelet grid incrementally (shape Nt x Nf, matching helper description)
    wave = np.zeros((Nt, Nf), dtype=float)

    # Precompute a "fully-filled" wave for consistent color scaling and correctness checks.
    wave_full = np.zeros((Nt, Nf), dtype=float)
    for fb, dx in enumerate(dx_packets):
        if fb == 0:
            wave_full[0:Nt:2, 0] = np.real(dx)[0:Nt:2] * np.sqrt(2.0)
        elif fb == Nf:
            wave_full[1:Nt:2, 0] = np.real(dx)[0:Nt:2] * np.sqrt(2.0)
        else:
            wave_full[:, fb] = _pack_wave_column_from_dx(dx, Nt, fb)

    # Match the public API scaling: from_freq_to_wavelet returns (2/Nf)*sqrt(2) * wave.T.
    factor = (2.0 / Nf) * np.sqrt(2.0)
    wavelet_full = factor * wave_full.T  # (Nf, Nt)
    z = np.abs(wavelet_full).ravel()
    z_pos = z[z > 0]
    vmin = float(np.percentile(z_pos, 5)) if z_pos.size else 1e-6
    vmax = float(np.percentile(z_pos, 99.5)) if z_pos.size else 1.0
    vmin = max(vmin, 1e-12)
    vmax = max(vmax, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Correctness check against library implementation (NumPy backend).
    f = np.fft.rfftfreq(Nf * Nt, d=dt)
    fs = FrequencySeries(data=yf, freq=f)
    lib_wave = from_freq_to_wavelet(fs, Nf=Nf, Nt=Nt, nx=nx)
    if not np.allclose(lib_wave.data, wavelet_full, rtol=5e-6, atol=5e-9):
        max_err = float(np.max(np.abs(np.asarray(lib_wave.data) - wavelet_full)))
        raise RuntimeError(f"freq->wavelet manual mismatch (max abs err={max_err:.3e})")

    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_packet = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, :])

    # Spectrum panel
    (spec_line,) = ax_spec.semilogy(
        f, power, color="#6e7781", alpha=0.6, label="spectrum"
    )
    ax_spec.set_title("One-sided spectrum + windowed band")
    ax_spec.set_xlabel("frequency [Hz]")
    ax_spec.set_ylabel("power")
    ax_spec.set_xlim(0, f[-1])

    band_span = ax_spec.axvspan(0.0, 0.0, color="#fb8c00", alpha=0.12)
    band_center = ax_spec.axvline(0.0, color="#fb8c00", alpha=0.6, linewidth=1)

    # Twin axis for the window amplitude (keeps scaling intuitive)
    ax_win = ax_spec.twinx()
    (win_line,) = ax_win.plot(
        f, np.zeros_like(f), color="#fb8c00", linewidth=2, label="|phitilde| (norm)"
    )
    ax_win.set_ylim(0.0, 1.05)
    ax_win.set_ylabel("window amplitude")
    ax_win.grid(False)
    ax_spec.legend(handles=[spec_line, win_line], frameon=False, loc="upper right")

    # Packet panel
    n = np.arange(Nt)
    (packet_r,) = ax_packet.plot(n, np.real(dx_packets[0]), label="Re", color="#0969da")
    (packet_i,) = ax_packet.plot(n, np.imag(dx_packets[0]), label="Im", color="#8250df", alpha=0.85)
    ax_packet.set_title("IFFT packet (per f-bin)")
    ax_packet.set_xlabel("time-bin index n")
    ax_packet.set_ylabel("amplitude")
    ax_packet.legend(frameon=False, loc="upper right")
    ax_packet.grid(True, alpha=0.2)

    # Grid panel
    grid_im = ax_grid.imshow(
        np.abs((factor * wave.T)) + 1e-12,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Reds",
        norm=norm,
    )
    ax_grid.set_title("Wavelet grid (fills by frequency bin)")
    ax_grid.set_xlabel("time bins (n)")
    ax_grid.set_ylabel("frequency bins (f_bin)")
    cbar = fig.colorbar(grid_im, ax=ax_grid, pad=0.01)
    cbar.set_label(r"$|W|$")

    fig.tight_layout()

    def _update(frame: int):
        nonlocal band_span, wave
        fb = frame

        # update spectrum band: indices around jj_base of width Nt//2
        jj_base = fb * (Nt // 2)
        start = max(jj_base - (Nt // 2) + 1, 0)
        end = min(jj_base + (Nt // 2), len(f) - 1)
        idx = np.arange(start, end + 1)

        # Highlight the active band on the x-axis.
        band_span.remove()
        band_span = ax_spec.axvspan(f[start], f[end], color="#fb8c00", alpha=0.12)
        band_center.set_xdata([f[jj_base], f[jj_base]])

        # Window curve on its own axis (normalized).
        j_idx = np.abs(np.arange(len(f)) - jj_base)
        j_idx = np.clip(j_idx, 0, len(phif) - 1)
        win_full = np.zeros_like(f, dtype=float)
        win_full[idx] = phif[j_idx[idx]]
        win_full = win_full / (np.max(phif) if np.max(phif) else 1.0)
        win_line.set_ydata(win_full)

        # update packet
        dx = dx_packets[fb]
        packet_r.set_ydata(np.real(dx))
        packet_i.set_ydata(np.imag(dx))
        ax_packet.set_ylim(
            min(np.min(np.real(dx)), np.min(np.imag(dx))) * 1.2,
            max(np.max(np.real(dx)), np.max(np.imag(dx))) * 1.2,
        )

        # update wave grid: write this f_bin's contribution
        if fb == 0:
            wave[0:Nt:2, 0] = np.real(dx)[0:Nt:2] * np.sqrt(2.0)
        elif fb == Nf:
            wave[1:Nt:2, 0] = np.real(dx)[0:Nt:2] * np.sqrt(2.0)
        else:
            wave[:, fb] = _pack_wave_column_from_dx(dx, Nt, fb)

        grid_im.set_data(np.abs((factor * wave.T)) + 1e-12)
        ax_grid.set_title(f"Wavelet grid (filling f_bin={fb}/{Nf})")
        return (packet_r, packet_i, grid_im)

    anim = animation.FuncAnimation(
        fig, _update, frames=Nf + 1, interval=140, blit=False, repeat=True
    )
    _save_gif(anim, "freq_to_wavelet.gif", fps=8)
    plt.close(fig)


# ----------------------------
# Time -> wavelet animation
# ----------------------------


def _windowed_packet_time(
    data: np.ndarray, phi: np.ndarray, Nf: int, Nt: int, mult: int, t_bin: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure-Python version of the core idea in
    src/pywavelet/transforms/numpy/forward/from_time.py.

    Returns:
      wdata: windowed segment (length K)
      wfft: rfft(wdata) (length K//2+1)
    """
    ND = Nf * Nt
    K = mult * 2 * Nf
    wdata = np.zeros(K, dtype=float)
    data_pad = np.concatenate([data, data[:K]])

    jj = (t_bin * Nf - K // 2) % ND
    for j in range(K):
        wdata[j] = data_pad[jj] * phi[j]
        jj = (jj + 1) % ND

    return wdata, np.fft.rfft(wdata, K)


def _pack_wave_row_from_rfft(wfft: np.ndarray, Nf: int, mult: int, t_bin: int) -> np.ndarray:
    """Mimic the compact packing in __fill_wave_2 for one time-bin row."""
    row = np.zeros(Nf, dtype=float)
    if t_bin % 2 == 0:
        row[0] = wfft[0].real / np.sqrt(2.0)
    # fill Cnm parity pattern
    for j in range(1, Nf):
        if (t_bin + j) % 2:
            row[j] = -wfft[j * mult].imag
        else:
            row[j] = wfft[j * mult].real
    return row


def make_time_to_wavelet_gif() -> None:
    Nf = 32
    Nt = 32
    dt = 1 / 256
    nx = 4.0
    mult = 8

    ts = _chirp_timeseries(dt, Nf=Nf, Nt=Nt)
    phi = phi_vec(Nf, d=nx, q=mult)
    phi = np.asarray(phi, dtype=float)

    ND = Nf * Nt
    K = mult * 2 * Nf

    # Precompute packets per time bin
    packets = [(_windowed_packet_time(ts.data, phi, Nf, Nt, mult, tb)) for tb in range(Nt)]

    wave = np.zeros((Nt, Nf), dtype=float)  # time bins x freq bins

    # Precompute a "fully-filled" wave (in the same scale as Wavelet.data).
    wave_full = np.zeros((Nt, Nf), dtype=float)
    for tb, (_wdata, wfft) in enumerate(packets):
        # columns j>=1 (scaled by sqrt2 at the public API level)
        packed = _pack_wave_row_from_rfft(wfft, Nf, mult, tb)
        wave_full[tb, 1:] = packed[1:] * np.sqrt(2.0)
        # special column 0 handling writes two time bins at once
        if tb % 2 == 0 and tb < Nt - 1:
            wave_full[tb, 0] = wfft[0].real
            wave_full[tb + 1, 0] = wfft[mult * Nf].real

    z = np.abs(wave_full.T).ravel()
    z_pos = z[z > 0]
    vmin = float(np.percentile(z_pos, 5)) if z_pos.size else 1e-6
    vmax = float(np.percentile(z_pos, 99.5)) if z_pos.size else 1.0
    vmin = max(vmin, 1e-12)
    vmax = max(vmax, vmin * 10)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Correctness check against library implementation (NumPy backend).
    lib_wave = from_time_to_wavelet(ts, Nf=Nf, Nt=Nt, nx=nx, mult=mult)
    if not np.allclose(lib_wave.data, wave_full.T, rtol=5e-6, atol=5e-9):
        max_err = float(np.max(np.abs(np.asarray(lib_wave.data) - wave_full.T)))
        raise RuntimeError(f"time->wavelet manual mismatch (max abs err={max_err:.3e})")

    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax_time = fig.add_subplot(gs[0, 0])
    ax_fft = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, :])

    t = np.arange(ND) * dt
    (time_line,) = ax_time.plot(t, ts.data, color="#6e7781", alpha=0.6)
    ax_time.set_title("Time series + sliding window")
    ax_time.set_xlabel("time [s]")
    ax_time.set_ylabel("amplitude")
    ax_time.set_xlim(t[0], t[-1])
    ax_time.grid(True, alpha=0.2)

    # moving window overlay (in time)
    win_patches = [ax_time.axvspan(t[0], t[0], color="#fb8c00", alpha=0.25)]

    # Overlay the actual window phi(t) (scaled) on the time-domain plot.
    # Note: phi is applied as wdata[j] = data[idx] * phi[j] in the implementation.
    phi_norm = phi / (np.max(np.abs(phi)) if np.max(np.abs(phi)) else 1.0)
    phi_scale = 0.9 * float(np.max(np.abs(ts.data)))
    phi_overlay = phi_norm * phi_scale
    (phi_line_1,) = ax_time.plot(
        [],
        [],
        color="#8250df",
        linewidth=2,
        alpha=0.55,
        linestyle="--",
        label=r"$\phi(t)$ (scaled)",
    )
    (phi_line_2,) = ax_time.plot(
        [],
        [],
        color="#8250df",
        linewidth=2,
        alpha=0.55,
        linestyle="--",
    )
    ax_time.legend(frameon=False, loc="upper right")

    # FFT panel
    freqs = np.fft.rfftfreq(K, d=dt)
    (fft_line,) = ax_fft.semilogy(freqs, np.abs(packets[0][1]) ** 2 + 1e-20, color="#0969da")
    ax_fft.set_title("RFFT of windowed segment")
    ax_fft.set_xlabel("frequency [Hz]")
    ax_fft.set_ylabel("power")
    ax_fft.set_xlim(0, freqs[-1])
    ax_fft.grid(True, alpha=0.2)

    # highlight sampled frequencies j*mult (plus the special DC/Nyquist used for column 0)
    sample_freqs = freqs[mult * np.arange(1, Nf)]
    sample_scatter = ax_fft.scatter(
        sample_freqs,
        np.ones_like(sample_freqs),
        s=18,
        color="#cf222e",
        alpha=0.7,
        label="samples (j*mult)",
    )
    special_scatter = ax_fft.scatter(
        [freqs[0], freqs[mult * Nf]],
        [1.0, 1.0],
        s=22,
        color="#fb8c00",
        alpha=0.9,
        label="DC / Nyquist (col 0)",
    )
    ax_fft.legend(frameon=False, loc="upper right")

    # Grid panel
    grid_im = ax_grid.imshow(
        np.abs(wave).T + 1e-12,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Reds",
        norm=norm,
    )
    ax_grid.set_title("Wavelet grid (fills by time bin)")
    ax_grid.set_xlabel("time bins (t_bin)")
    ax_grid.set_ylabel("frequency bins")
    cbar = fig.colorbar(grid_im, ax=ax_grid, pad=0.01)
    cbar.set_label(r"$|W|$")

    fig.tight_layout()

    def _update(frame: int):
        nonlocal win_patches, wave
        tb = frame
        wdata, wfft = packets[tb]

        # window overlay bounds: center it at tb*Nf with width K
        center = tb * Nf
        start = (center - K // 2) * dt
        end = (center + K // 2) * dt
        T = ND * dt

        # remove previous patches
        for p in win_patches:
            p.remove()
        win_patches = []

        # show periodic wrap by splitting into up to 2 spans within [0, T]
        start_mod = start % T
        end_mod = end % T
        if start_mod <= end_mod:
            win_patches.append(ax_time.axvspan(start_mod, end_mod, color="#fb8c00", alpha=0.25))
        else:
            win_patches.append(ax_time.axvspan(0.0, end_mod, color="#fb8c00", alpha=0.25))
            win_patches.append(ax_time.axvspan(start_mod, T, color="#fb8c00", alpha=0.25))

        # update phi overlay to match the exact indices used (with periodic wrapping)
        jj0 = (tb * Nf - K // 2) % ND
        idx = (jj0 + np.arange(K)) % ND
        t_phi = t[idx]
        # split into two monotonic segments for plotting
        breaks = np.where(np.diff(t_phi) < 0)[0]
        if breaks.size:
            b = int(breaks[0]) + 1
            phi_line_1.set_data(t_phi[:b], phi_overlay[:b])
            phi_line_2.set_data(t_phi[b:], phi_overlay[b:])
        else:
            phi_line_1.set_data(t_phi, phi_overlay)
            phi_line_2.set_data([], [])

        # update fft curve
        pow_ = np.abs(wfft) ** 2 + 1e-20
        fft_line.set_ydata(pow_)
        ax_fft.set_ylim(max(np.min(pow_), 1e-18), np.max(pow_) * 1.2)

        # update sampled points
        y_samples = pow_[mult * np.arange(1, Nf)]
        sample_scatter.set_offsets(np.c_[sample_freqs, y_samples])
        special_scatter.set_offsets(
            np.c_[[freqs[0], freqs[mult * Nf]], [pow_[0], pow_[mult * Nf]]]
        )

        # pack wave row:
        # - columns j>=1 are filled for every tb (and scaled by sqrt2 in the public API)
        packed = _pack_wave_row_from_rfft(wfft, Nf, mult, tb)
        wave[tb, 1:] = packed[1:] * np.sqrt(2.0)
        # - column 0 is special: it is written for tb even, and also writes tb+1
        if tb % 2 == 0 and tb < Nt - 1:
            wave[tb, 0] = wfft[0].real
            wave[tb + 1, 0] = wfft[mult * Nf].real

        grid_im.set_data(np.abs(wave).T + 1e-12)
        ax_grid.set_title(f"Wavelet grid (filling t_bin={tb}/{Nt-1})")
        return (fft_line, grid_im)

    anim = animation.FuncAnimation(
        fig, _update, frames=Nt, interval=140, blit=False, repeat=True
    )
    _save_gif(anim, "time_to_wavelet.gif", fps=8)
    plt.close(fig)


def main() -> None:
    make_time_to_wavelet_gif()
    make_freq_to_wavelet_gif()


if __name__ == "__main__":
    main()
