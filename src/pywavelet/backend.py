import importlib
import os

from .logger import logger

JAX = "jax"
CUPY = "cupy"
NUMPY = "numpy"


def get_backend_from_env():
    """Select and return the appropriate backend module."""
    backend = os.getenv("PYWAVELET_BACKEND", NUMPY).lower()

    if backend == JAX:
        if importlib.util.find_spec(JAX):
            import jax.numpy as xp
            from jax.numpy.fft import fft, ifft, irfft, rfft, rfftfreq
            from jax.scipy.special import betainc

            logger.info("Using JAX backend")
            return xp, fft, ifft, irfft, rfft, rfftfreq, betainc, backend
        else:
            logger.warning(
                "JAX backend requested but not installed. Falling back to NumPy."
            )

    elif backend == CUPY:
        if importlib.util.find_spec(CUPY):
            import cupy as xp
            from cupy.fft import fft, ifft, irfft, rfft, rfftfreq
            from cupyx.scipy.special import betainc

            logger.info("Using CuPy backend")
            return xp, fft, ifft, irfft, rfft, rfftfreq, betainc, backend
        else:
            logger.warning(
                "CuPy backend requested but not installed. Falling back to NumPy."
            )

    # Default to NumPy
    import numpy as xp
    from numpy.fft import fft, ifft, irfft, rfft, rfftfreq
    from scipy.special import betainc

    logger.info("Using NumPy+Numba backend")
    return xp, fft, ifft, irfft, rfft, rfftfreq, betainc, backend


# Get the chosen backend
xp, fft, ifft, irfft, rfft, rfftfreq, betainc, current_backend = (
    get_backend_from_env()
)
