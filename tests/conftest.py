import os

import pytest
import numpy as np
from utils import (
    generate_chirp_time_domain_signal,
    generate_sine_time_domain_signal,
    BRANCH
)

# set global env var "NUMBA_DISABLE_JIT=1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = f"{HERE}/test_data"

fs = 1024
fmax = 50
frange = [10, fmax]
dt = 1 / fs
Nt = 2 ** 6
Nf = 2 ** 7
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
    return generate_sine_time_domain_signal(ts, ND, f_true=10).to_frequencyseries()
