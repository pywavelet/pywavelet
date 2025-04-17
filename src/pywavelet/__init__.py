"""
WDM Wavelet transform
"""

import importlib
import os

from . import backend as _backend
from ._version import __version__


def set_backend(backend: str, precision: str = "float32") -> None:
    """Set the backend for the wavelet transform.

    Parameters
    ----------
    backend : str
        Backend to use. Options are "numpy", "jax", "cupy".
    """
    from . import transforms, types

    os.environ["PYWAVELET_BACKEND"] = backend
    os.environ["PYWAVELET_PRECISION"] = precision

    importlib.reload(_backend)
    importlib.reload(types)
    importlib.reload(transforms)
    if backend == "cupy":
        importlib.reload(transforms.cupy)
        importlib.reload(transforms.cupy.forward)
        importlib.reload(transforms.cupy.inverse)
    elif backend == "jax":
        importlib.reload(transforms.jax)
        importlib.reload(transforms.jax.forward)
        importlib.reload(transforms.jax.inverse)
    else:
        importlib.reload(transforms.numpy)
        importlib.reload(transforms.numpy.forward)
        importlib.reload(transforms.numpy.inverse)
