import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from utils import cuda_available

from pywavelet import set_backend
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd


@pytest.mark.parametrize("backend", ["jax", "cupy", "numpy"])
def test_backend_loader(backend):

    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")
    set_backend(backend)
    from pywavelet.backend import current_backend, xp

    assert current_backend == backend

    set_backend("numpy")


def test_backend_fails_if_no_cupy():
    if cuda_available:
        pytest.skip("CUDA is available")

    # assert AttributeError is raised when trying to import cupy
    with pytest.raises(AttributeError):
        set_backend("cupy")


def test_backed_logger():
    from pywavelet.backend import get_available_backends_table

    print(get_available_backends_table())
