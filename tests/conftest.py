import os

import numpy as np
import pytest
from utils import (
    BRANCH,
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
)

from pywavelet.types import Wavelet
from pywavelet.types.wavelet_bins import compute_bins

# set global env var "NUMBA_DISABLE_JIT=1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{HERE}/test_data"

fs = 1024
fmax = 50
frange = [10, fmax]
dt = 1 / fs
Nt = 2**6
Nf = 2**7
mult = 16
ND = Nt * Nf
ts = np.arange(0, ND) * dt


@pytest.fixture()
def plot_dir():
    dirname = f"{HERE}/out_plots_{BRANCH}"
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture()
def chirp_time():
    return generate_chirp_time_domain_signal(ts, frange)


@pytest.fixture()
def chirp_freq():
    return generate_chirp_time_domain_signal(ts, frange).to_frequencyseries()


@pytest.fixture()
def sine_time():
    return generate_sine_time_domain_signal(ts, ND, f_true=10)


@pytest.fixture()
def sine_freq():
    return generate_sine_time_domain_signal(
        ts, ND, f_true=10
    ).to_frequencyseries()


def monochromatic_wnm(
    f0: float = 20,
    dt: float = 0.0125,
    A: float = 2,
    Nt: int = 128,
    Nf: int = 256,
):
    T = Nt * Nf * dt
    N = Nt * Nf
    t_bins, f_bins = compute_bins(Nf, Nt, T)
    wnm = np.zeros((Nt, Nf))
    m0 = int(f0 * N * dt)
    f0_bin_idx = int(2 * m0 / Nt)
    odd_t_indices = np.arange(Nt) % 2 != 0
    wnm[odd_t_indices, f0_bin_idx] = A * np.sqrt(2 * Nf)
    return Wavelet(wnm.T, t_bins, f_bins)
