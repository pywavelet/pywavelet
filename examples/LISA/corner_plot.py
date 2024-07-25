import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from matplotlib.font_manager import FontProperties

os.chdir(
    "/Users/alexander_burke/Documents/LISA_Science/Projects/Noise/pywavelet/examples/LISA/data/wavelet"
)
samples_wavelet = np.load("samples.npy")
os.chdir(
    "/Users/alexander_burke/Documents/LISA_Science/Projects/Noise/pywavelet/examples/LISA/data/freq_domain"
)
samples_FD = np.load("samples.npy")

# Wavelet domain first
params = [r"$\log_{10}a$", r"$\log_{10}\dot{f}$", r"$\log_{10}\ddot{f}$"]
N_param = len(params)

N_samples_wavelets = len(samples_wavelet[0])
N_samples_FD = len(samples_FD[0])

weights = 0.8 * (N_samples_wavelets / N_samples_FD) * np.ones(N_samples_FD)
samples_wavelet_stack = np.column_stack(samples_wavelet)
samples_FD = np.column_stack(samples_FD)


corner_kwrgs = dict(
    plot_datapoints=False,
    smooth1d=True,
    labels=params,
    label_kwargs={"fontsize": 12},
    set_xlabel={"fontsize": 20},
    show_titles=True,
    title_fmt=".7f",
    title_kwargs={"fontsize": 9},
    smooth=True,
)

figure = corner(samples_wavelet_stack, bins=30, color="blue", **corner_kwrgs)

corner(
    samples_FD,
    bins=30,
    fig=figure,
    weights=weights,
    color="red",
    **corner_kwrgs,
)
axes = np.array(figure.axes).reshape((N_param, N_param))

true_vals = np.log10(np.array([5e-21, 1e-3, 1e-8]))

for k in range(N_param):
    ax = axes[k, k]
    ax.axvline(true_vals[k], color="g")

for yi in range(N_param):
    for xi in range(yi):
        ax = axes[yi, xi]

        ax.axhline(true_vals[yi], color="g")
        ax.axvline(true_vals[xi], color="g")

        ax.plot(true_vals[xi], true_vals[yi], "sg")

for ax in figure.get_axes():
    ax.tick_params(axis="both", labelsize=8)

blue_line = mlines.Line2D([], [], color="blue", label="Wavelet Domain")
red_line = mlines.Line2D([], [], color="red", label="Frequency Domain")

plt.legend(
    handles=[blue_line, red_line],
    fontsize=15,
    frameon=True,
    bbox_to_anchor=(0.5, N_param),
    loc="upper right",
    title=r"Domain Comparison",
    title_fontproperties=FontProperties(size=15, weight="bold"),
)

plt.tight_layout()
plt.savefig("post_wavelet_FD.pdf", bbox_inches="tight")
plt.show()
plt.clf()
plt.close()
