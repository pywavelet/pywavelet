from ...logger import logger
from .forward import from_freq_to_wavelet, from_time_to_wavelet
from .inverse import from_wavelet_to_freq, from_wavelet_to_time

logger.warning("JAX SUBPACKAGE NOT FULLY TESTED")


def _log_jax_info():
    """Log JAX backend and precision information.

    backend : str
        JAX backend. ["cpu", "gpu", "tpu"]
    precision : str
        JAX precision. ["32bit", "64bit"]
    """
    import jax

    _backend = jax.default_backend()
    _precision = "64bit" if jax.config.jax_enable_x64 else "32bit"

    logger.info(f"Jax running on {_backend} [{_precision} precision].")
    if _precision == "32bit":
        logger.warning(
            "Jax is not running in 64bit precision. "
            "To change, use jax.config.update('jax_enable_x64', True)."
        )


_log_jax_info()

__all__ = [
    "from_wavelet_to_time",
    "from_wavelet_to_freq",
    "from_time_to_wavelet",
    "from_freq_to_wavelet",
]
