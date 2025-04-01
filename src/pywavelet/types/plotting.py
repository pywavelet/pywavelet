import h5py
import numpy as np

from lisatools.sensitivity import get_sensitivity

from pywavelet.types import FrequencySeries, Wavelet
from dataclasses import dataclass
from typing import Dict
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

import matplotlib.pyplot as plt

# Set the desired RC parameters
rc_params = {
    'xtick.direction': 'in',  # Mirrored ticks (in and out)
    'ytick.direction': 'in',
    'xtick.top': True,  # Show ticks on the top spine
    'ytick.right': True,  # Show ticks on the right spine
    'xtick.major.size': 6,  # Length of major ticks
    'ytick.major.size': 6,
    'xtick.minor.size': 4,  # Length of minor ticks
    'ytick.minor.size': 4,
    'xtick.major.pad': 4,  # Padding between tick and label
    'ytick.major.pad': 4,
    'xtick.minor.pad': 4,
    'ytick.minor.pad': 4,
    'font.size': 14,  # Overall font size
    'axes.labelsize': 16,  # Font size of axis labels
    'axes.titlesize': 18,  # Font size of plot title
    'xtick.labelsize': 12,  # Font size of x-axis tick labels
    'ytick.labelsize': 12,  # Font size of y-axis tick labels
    'xtick.major.width': 2,  # Thickness of major ticks
    'ytick.major.width': 2,  # Thickness of major ticks
    'xtick.minor.width': 1,  # Thickness of minor ticks
    'ytick.minor.width': 1,  # Thickness of minor ticks
    'lines.linewidth': 3,  # Default linewidth for lines in plots
    'patch.linewidth': 4,  # Default linewidth for patches (e.g., rectangles, circles)
    'axes.linewidth': 2  # Default linewidth for the axes spines

}

# Apply the RC parameters globally
plt.rcParams.update(rc_params)

COLORS = dict(
    mbh="tab:purple",
    emri="tab:red",
    gb="tab:cyan"
)


@dataclass
class PlotData:
    freqseries: FrequencySeries
    wavelet: Wavelet
    label: str


def load_data() -> Dict[str, PlotData]:
    """
    Load data from HDF5 files.
    Returns:
        Tuple[np.ndarray, np.ndarray]: (frequency array, waveform)
    """
    # Load the data from the HDF5 file
    keys = ['emri', 'gb', 'mbh']
    data = {}
    with h5py.File('data.h5', 'r') as f:
        for key in keys:
            freqseries = FrequencySeries(f[key]['hf'][:], f[key]['freq'][:])
            # if key == 'gb':
            #     ts = freqseries.to_timeseries()
            #     ts = ts * scipy.signal.windows.tukey(len(ts), alpha=0.1)
            #     freqseries = ts.to_frequencyseries()

            wdm = freqseries.to_wavelet(Nf=1024)
            data[key] = PlotData(freqseries=freqseries, wavelet=wdm, label=key.upper())
    return data


def generate_plot(data: Dict[str, PlotData]):
    # Create figure and gridspec
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])  # 2 rows, 3 columns

    # Create subplots
    ax0 = fig.add_subplot(gs[0, :])  # Row 1: spans all 3 columns
    ax1 = fig.add_subplot(gs[1, 0])  # Row 2, Column 1
    ax2 = fig.add_subplot(gs[1, 1])  # Row 2, Column 2
    ax3 = fig.add_subplot(gs[1, 2])  # Row 2, Column 3
    axs = [ax1, ax2, ax3]

    # rearrange data [emri, mbh, gb]
    data = {k: data[k] for k in ['emri', 'mbh', 'gb']}

    ### Plot characteristic strain
    f_fixed = np.logspace(-4, -1, 1000)
    sensitivity = get_sensitivity(f_fixed, sens_fn="LISASens", return_type="char_strain")
    for k, d in data.items():
        f, hf = d.freqseries.freq, d.freqseries.data
        zorder=-10 if k=='emri' else 10
        ax0.loglog(f[1:], (np.abs(hf) * f)[1:], label=f'{d.label}', color=COLORS[k], zorder=zorder)
    ax0.loglog(f_fixed, sensitivity, 'k--', label='LISA Sensitivity', zorder=-5)
    ax0.legend(frameon=False)
    ax0.set_xlim(1e-4, 1e-1)
    ax0.set_ylim(bottom=10 ** - 22, top=10 ** -18)
    # reduce padding for xlabel
    ax0.set_xlabel(r'$f$ [Hz]', labelpad=-5)
    ax0.set_ylabel(r"Characteristic Strain")

    ### Plot WDMs (share colorbar)
    wdm_range = (np.inf, -np.inf)
    for d in data.values():
        wdm = np.abs(d.wavelet.data)
        wdm_range = (min(wdm_range[0], np.min(wdm)), max(wdm_range[1], np.max(wdm)))
    # log_norm = LogNorm(vmin=wdm_range[0], vmax=wdm_range[1])
    log_norm = LogNorm(vmin=10 ** -31, vmax=10 ** -19)

    # set text color = white



    for i, (k, d) in enumerate(data.items()):
        kwgs = dict(
            absolute=True, zscale='log', cmap='inferno',
            ax=axs[i], norm=log_norm,
            show_gridinfo=False,
            show_colorbar=False,
        )
        if i == 2:
            kwgs['show_colorbar'] = True
        d.wavelet.plot(**kwgs)
        axs[i].text(
            0.05,
            0.95,
            d.label,
            color=COLORS[k],
            transform=axs[i].transAxes,
            verticalalignment="top",
            # extra bold font
            fontweight='bold',
        )

    ax1.set_ylabel(r"$f$ [Hz]")
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax1.set_ylim(top=0.02)

    cbar = plt.gca().images[-1].colorbar

    # cbar = ax3.collections[0].colorbar
    cbar.set_label(r'')
    cbar.ax.yaxis.label.set(rotation=0, ha='right', va='bottom')
    cbar.ax.yaxis.set_tick_params(rotation=0)

    # make yticks only use 2 digits, and use only 3 ticks
    for ax in axs:
        ax.set_xlabel(r"$t$ [days]")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    return fig, axs


# data = load_data()
fig, ax = generate_plot(data)
plt.tight_layout()
