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
    from . import types
    from . import transforms
    os.environ["PYWAVELET_BACKEND"] = backend
    os.environ["PYWAVELET_PRECISION"] = precision

    importlib.reload(_backend)
    importlib.reload(types)
    importlib.reload(transforms)
