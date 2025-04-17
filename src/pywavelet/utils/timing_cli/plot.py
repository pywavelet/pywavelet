import glob
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot_runtimes"]


def _cache_all_runtimes(outdir: str):
    data = {}
    for f in glob.glob(os.path.join(outdir, "runtime_*.csv")):
        df = pd.read_csv(f)
        label = f.split("runtime_")[1].split(".")[0]
        data[label] = df.to_dict(orient="records")

    cache_fn = os.path.join(outdir, "runtimes.json")
    # load any existing data
    if os.path.exists(cache_fn):
        with open(cache_fn, "r") as f:
            existing_data = json.load(f)
            data.update(existing_data)

    # save to json
    with open(cache_fn, "w") as f:
        json.dump(data, f, indent=4)

    return cache_fn


def plot_runtimes(outdir: str):
    cache_fn = _cache_all_runtimes(outdir)

    fig, ax = plt.subplots(figsize=(4, 3.5))

    with open(cache_fn, "r") as f:
        data = json.load(f)
        for label, runtimes in data.items():
            runtimes = pd.DataFrame(runtimes)
            _plot(runtimes, ax=ax, label=label)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "runtimes.png"), bbox_inches="tight")


def _plot(
    runtimes: pd.DataFrame, ax=None, **kwgs
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.5))
    fig = ax.figure

    runtimes = runtimes.dropna()
    runtimes = runtimes.sort_values(by="ND")

    nds = runtimes["ND"].values
    times, stds = runtimes["median"], runtimes["std"]
    ax.plot(nds, times, **kwgs)
    kwgs["label"] = None
    ax.fill_between(
        nds,
        np.array(times) - np.array(stds),
        np.array(times) + np.array(stds),
        alpha=0.3,
        **kwgs,
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Data Points")
    ax.set_ylabel("Runtime (s)")
    return fig, ax
