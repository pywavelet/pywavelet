import os

from .logger import logger

try:
    import jax

    jax_available = True


except ImportError:
    jax_available = False

use_jax = jax_available and os.getenv("PYWAVELET_JAX", "0") == "1"

if use_jax:
    import jax.numpy as xp  # type: ignore
    from jax.numpy.fft import fft, ifft, irfft, rfft, rfftfreq  # type: ignore
    from jax.scipy.special import betainc  # type: ignore

    logger.info("Using JAX backend")

else:
    import numpy as xp  # type: ignore
    from numpy.fft import fft, ifft, irfft, rfft, rfftfreq  # type: ignore
    from scipy.special import betainc  # type: ignore

    logger.info("Using NumPy+numba backend")


PI = xp.pi
