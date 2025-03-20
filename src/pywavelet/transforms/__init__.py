from ..backend import current_backend

if current_backend == "jax":
    from .jax import (
        from_freq_to_wavelet,
        from_time_to_wavelet,
        from_wavelet_to_freq,
        from_wavelet_to_time,
    )
elif current_backend == "cupy":
    from .cupy import (
        from_freq_to_wavelet,
        from_time_to_wavelet,
        from_wavelet_to_freq,
        from_wavelet_to_time,
    )
else:
    from .numpy import (
        from_wavelet_to_time,
        from_wavelet_to_freq,
        from_time_to_wavelet,
        from_freq_to_wavelet,
    )

from .phi_computer import omega, phi_vec, phitilde_vec_norm

__all__ = [
    "from_wavelet_to_time",
    "from_wavelet_to_freq",
    "from_time_to_wavelet",
    "from_freq_to_wavelet",
    "phitilde_vec_norm",
    "phi_vec",
]
