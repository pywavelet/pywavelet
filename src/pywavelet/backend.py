import importlib
import os
from typing import Tuple

import numpy as np
from rich.console import Console
from rich.table import Table, Text

from .logger import logger

JAX = "jax"
CUPY = "cupy"
NUMPY = "numpy"

VALID_PRECISIONS = ["float32", "float64"]


def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    # Check if CuPy is available and CUDA is accessible
    cupy_available = importlib.util.find_spec("cupy") is not None
    _cuda_available = False
    if cupy_available:
        import cupy

        try:
            cupy.cuda.runtime.getDeviceCount()  # Check if any CUDA device is available
            _cuda_available = True
        except cupy.cuda.runtime.CUDARuntimeError:
            _cuda_available = False
    else:
        _cuda_available = False
    return _cuda_available


def jax_is_available() -> bool:
    """Check if JAX is available."""
    return importlib.util.find_spec(JAX) is not None


def get_available_backends_table():
    """Print the available backends as a rich table."""

    jax_avail = jax_is_available()
    cupy_avail = cuda_is_available()
    table = Table("Backend", "Available", title="Available backends")
    true_check = "[green]✓[/green]"
    false_check = "[red]✗[/red]"
    table.add_row(JAX, true_check if jax_avail else false_check)
    table.add_row(CUPY, true_check if cupy_avail else false_check)
    table.add_row(NUMPY, true_check)
    console = Console(width=150)
    with console.capture() as capture:
        console.print(table)
    return Text.from_ansi(capture.get())


def log_backend(level="info"):
    """Print the current backend and precision."""
    backend = os.getenv("PYWAVELET_BACKEND", NUMPY).lower()
    precision = os.getenv("PYWAVELET_PRECISION", "float32").lower()
    str = f"Current backend: {backend}[{precision}]"
    if level == "info":
        logger.info(str)
    elif level == "debug":
        logger.debug(str)


def get_backend_from_env():
    """Select and return the appropriate backend module."""
    backend = os.getenv("PYWAVELET_BACKEND", NUMPY).lower()
    precision = os.getenv("PYWAVELET_PRECISION", "float32").lower()

    if backend == JAX and jax_is_available():

        import jax.numpy as xp
        from jax.numpy.fft import fft, ifft, irfft, rfft, rfftfreq
        from jax.scipy.special import betainc

    elif backend == CUPY and cuda_is_available():

        import cupy as xp
        from cupy.fft import fft, ifft, irfft, rfft, rfftfreq
        from cupyx.scipy.special import betainc

    elif backend == NUMPY:
        import numpy as xp
        from numpy.fft import fft, ifft, irfft, rfft, rfftfreq
        from scipy.special import betainc

    else:
        logger.error(f"Backend {backend}[{precision}] is not available. ")
        print(get_available_backends_table())
        logger.warning(f"Setting backend to NumPy. ")
        os.environ["PYWAVELET_BACKEND"] = NUMPY
        return get_backend_from_env()

    log_backend("debug")
    return xp, fft, ifft, irfft, rfft, rfftfreq, betainc, backend


def get_precision_from_env() -> str:
    """Get the precision from the environment variable."""
    precision = os.getenv("PYWAVELET_PRECISION", "float32").lower()
    if precision not in VALID_PRECISIONS:
        logger.error(
            f"Precision {precision} is not supported, defaulting to float32."
        )
        precision = "float32"
    return precision


def set_precision(precision: str) -> None:
    """Set the precision for the backend."""
    precision = precision.lower()
    if precision not in VALID_PRECISIONS:
        logger.error(f"Precision {precision} is not supported.")
        return
    else:
        os.environ["PYWAVELET_PRECISION"] = precision
        logger.info(f"Setting precision to {precision}.")
        return


def get_dtype_from_env() -> Tuple[np.dtype, np.dtype]:
    """Get the data type from the environment variable."""
    precision = get_precision_from_env()
    backend = get_backend_from_env()[-1]
    if backend == JAX:

        if precision == "float32":
            import jax

            jax.config.update("jax_enable_x64", False)

            import jax.numpy as jnp

            float_dtype = jnp.float32
            complex_dtype = jnp.complex64
        elif precision == "float64":
            import jax

            jax.config.update("jax_enable_x64", True)

            import jax.numpy as jnp

            float_dtype = jnp.float64
            complex_dtype = jnp.complex128

    elif backend == CUPY:
        import cupy as cp

        if precision == "float32":
            float_dtype = cp.float32
            complex_dtype = cp.complex64
        elif precision == "float64":
            float_dtype = cp.float64
            complex_dtype = cp.complex128

    else:
        if precision == "float32":
            float_dtype = np.float32
            complex_dtype = np.complex64
        elif precision == "float64":
            float_dtype = np.float64
            complex_dtype = np.complex128

    return float_dtype, complex_dtype


cuda_available = cuda_is_available()

# Get the chosen backend
xp, fft, ifft, irfft, rfft, rfftfreq, betainc, current_backend = (
    get_backend_from_env()
)

# Get the chosen precision
float_dtype, complex_dtype = get_dtype_from_env()
