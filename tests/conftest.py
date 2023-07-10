import pytest
import os


@pytest.fixture()
def make_plots():
    return True

@pytest.fixture()
def plot_dir():
    dirname = 'out_plots'
    os.makedirs(dirname, exist_ok=True)
    return dirname