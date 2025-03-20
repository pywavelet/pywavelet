"""
WDM Wavelet transform
"""

import importlib
import os

from . import backend as _backend

__version__ = "0.0.2"


def set_backend(backend: str):
    """Set the backend for the wavelet transform.

    Parameters
    ----------
    backend : str
        Backend to use. Options are "numpy", "jax", "cupy".
    """
    from . import types
    from . import transforms
    os.environ["PYWAVELET_BACKEND"] = backend

    importlib.reload(_backend)
    importlib.reload(types)
    importlib.reload(transforms)
