import os

import jax
import numpy as np
import pytest
from utils import (
    BRANCH,
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
)

from pywavelet.logger import logger
from pywavelet.types import Wavelet
from pywavelet.types.wavelet_bins import compute_bins

jax.config.update("jax_enable_x64", True)

logger.setLevel("DEBUG")

# set global env var "NUMBA_DISABLE_JIT=1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{HERE}/test_data"

fmax = 35.0
frange = [10, fmax]
f0 = 20
dt = 0.0125
fs = 1 / dt
A = 2
Nt = 128
Nf = 256
ND = Nt * Nf
ts = np.arange(0, ND) * dt


def _gen_testdir():
    dirname = f"{HERE}/out_plots_{BRANCH}"
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture()
def plot_dir():
    return _gen_testdir()


@pytest.fixture()
def outdir():
    return _gen_testdir()


@pytest.fixture()
def chirp_time():
    return generate_chirp_time_domain_signal(ts, frange)


@pytest.fixture()
def chirp_freq():
    return generate_chirp_time_domain_signal(ts, frange).to_frequencyseries()


@pytest.fixture()
def sine_time():
    return generate_sine_time_domain_signal(ts, f_true=f0)


@pytest.fixture()
def sine_freq():
    return generate_sine_time_domain_signal(ts, f_true=f0).to_frequencyseries()


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
