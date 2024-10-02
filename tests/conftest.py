import os

import pytest
import numpy as np
from utils import generate_chirp_time_domain_signal, generate_sine_time_domain_signal
import subprocess

# set global env var "NUMBA_DISABLE_JIT=1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

HERE = os.path.abspath(os.path.dirname(__file__))

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
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
    dirname = f"{HERE}/out_plots_{branch_name}"
    os.makedirs(dirname, exist_ok=True)
    return dirname


@pytest.fixture()
def chirp_time():
    ht = generate_chirp_time_domain_signal(ts, frange)
    return ht


@pytest.fixture()
def chirp_freq():
    ht = generate_chirp_time_domain_signal(ts, frange)
    hf = ht.to_frequencyseries()
    return hf

@pytest.fixture()
def sine_time():
    ht = generate_sine_time_domain_signal(ts, ND, f_true=10)
    return ht

@pytest.fixture()
def sine_freq():
    ht = generate_sine_time_domain_signal(ts, ND, f_true=10)
    hf = ht.to_frequencyseries()
    return hf
