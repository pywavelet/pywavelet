import importlib
import os
from rich.table import Table, Text
from rich.console import Console



from .logger import logger

JAX = "jax"
CUPY = "cupy"
NUMPY = "numpy"


def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    # Check if CuPy is available and CUDA is accessible
    cupy_available = importlib.util.find_spec("cupy") is not None
    if cupy_available:
        import cupy

        try:
            cupy.cuda.runtime.getDeviceCount()  # Check if any CUDA device is available
            cuda_available = True
        except cupy.cuda.runtime.CUDARuntimeError:
            cuda_available = False
    else:
        cuda_available = False
    return cuda_available


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


def get_backend_from_env():
    """Select and return the appropriate backend module."""
    backend = os.getenv("PYWAVELET_BACKEND", NUMPY).lower()

    if backend == JAX and jax_is_available():

        import jax.numpy as xp
        from jax.numpy.fft import fft, ifft, irfft, rfft, rfftfreq
        from jax.scipy.special import betainc

        logger.info("Using JAX backend")

    elif backend == CUPY and cuda_is_available():

        import cupy as xp
        from cupy.fft import fft, ifft, irfft, rfft, rfftfreq
        from cupyx.scipy.special import betainc

        logger.info("Using CuPy backend")

    elif backend == NUMPY:
        import numpy as xp
        from numpy.fft import fft, ifft, irfft, rfft, rfftfreq
        from scipy.special import betainc

        logger.info("Using NumPy backend")


    else:
        logger.error(
            f"Backend {backend} is not available. "
        )
        print(get_available_backends_table())
        logger.warning(
            f"Setting backend to NumPy. "
        )
        os.environ["PYWAVELET_BACKEND"] = NUMPY
        return get_backend_from_env()

    return xp, fft, ifft, irfft, rfft, rfftfreq, betainc, backend


cuda_available = cuda_is_available()

# Get the chosen backend
xp, fft, ifft, irfft, rfft, rfftfreq, betainc, current_backend = (
    get_backend_from_env()
)
