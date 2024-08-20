import os

import pytest

# set global env var "NUMBA_DISABLE_JIT=1"
os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture()
def make_plots():
    return True


@pytest.fixture()
def plot_dir():
    dirname = "out_plots"
    os.makedirs(dirname, exist_ok=True)
    return dirname
