import importlib
import os

from .logger import logger


def get_backend():
    """Select and return the appropriate backend module."""
    backend = os.getenv("PYWAVELET_BACKEND", "numpy").lower()

    if backend == "jax":
        if importlib.util.find_spec("jax"):
            import jax
            import jax.numpy as xp
            from jax.numpy.fft import fft, ifft, irfft, rfft, rfftfreq
            from jax.scipy.special import betainc

            logger.info("Using JAX backend")
            return xp, fft, ifft, irfft, rfft, rfftfreq, betainc
        else:
            logger.warning(
                "JAX backend requested but not installed. Falling back to NumPy."
            )

    elif backend == "cupy":
        if importlib.util.find_spec("cupy"):
            import cupy as xp
            from cupy.fft import fft, ifft, irfft, rfft, rfftfreq
            from cupyx.scipy.special import betainc

            logger.info("Using CuPy backend")
            return xp, fft, ifft, irfft, rfft, rfftfreq, betainc
        else:
            logger.warning(
                "CuPy backend requested but not installed. Falling back to NumPy."
            )

    # Default to NumPy
    import numpy as xp
    from numpy.fft import fft, ifft, irfft, rfft, rfftfreq
    from scipy.special import betainc

    logger.info("Using NumPy+Numba backend")
    return xp, fft, ifft, irfft, rfft, rfftfreq, betainc


# Get the chosen backend
xp, fft, ifft, irfft, rfft, rfftfreq, betainc = get_backend()

# Constants
PI = xp.pi
