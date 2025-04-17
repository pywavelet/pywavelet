import importlib
import os

import numpy as np
import pytest
from conftest import DATA_DIR, Nf, Nt, dt

import pywavelet
import pywavelet.transforms
from pywavelet import set_backend
from pywavelet.backend import cuda_available, float_dtype, get_dtype_from_env


def get_array_type(backend):
    if backend == "cupy":
        import cupy as cp

        return cp.ndarray
    elif backend == "jax":
        import jax.numpy as jnp

        return jnp.ndarray
    else:
        return np.ndarray


def reload(backend):
    importlib.reload(pywavelet.backend)
    importlib.reload(pywavelet)
    importlib.reload(pywavelet.transforms)

    if backend == "cupy":
        importlib.reload(pywavelet.transforms.cupy)
    if backend == "jax":
        importlib.reload(pywavelet.transforms.jax)
        importlib.reload(pywavelet.transforms.jax.forward)
        importlib.reload(pywavelet.transforms.jax.inverse)
    importlib.reload(pywavelet.transforms.numpy)
    importlib.reload(pywavelet.transforms.numpy.forward)
    importlib.reload(pywavelet.transforms.numpy.inverse)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "jax"])
def test_np_precision(backend, sine_freq):

    precisions = ["float64", "float32"]
    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")

    for precision in precisions:
        print(
            "\n>>Testing backend: ",
            backend,
            " with precision: ",
            precision,
            " <<\n",
        )
        set_backend(backend, precision)
        reload(backend)

        assert os.getenv("PYWAVELET_PRECISION") == precision
        print(
            "expected dtypes: ",
            pywavelet.backend.float_dtype,
            pywavelet.backend.complex_dtype,
        )

        w = pywavelet.transforms.from_freq_to_wavelet(
            sine_freq,
            Nf=Nf,
            Nt=Nt,
        )

        float_dtype, complex_dtype = get_dtype_from_env()

        assert isinstance(w.data, get_array_type(backend))
        assert (
            w.data.dtype == float_dtype
        ), f"Expected {float_dtype}, but data.dtype->{w.data.dtype}"
