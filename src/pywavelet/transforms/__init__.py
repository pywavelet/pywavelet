from ..backend import use_jax

if use_jax:
    from .jax import (
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

from .phi_computer import phi_vec, phitilde_vec, phitilde_vec_norm

__all__ = [
    "from_wavelet_to_time",
    "from_wavelet_to_freq",
    "from_time_to_wavelet",
    "from_freq_to_wavelet",
    "phitilde_vec_norm",
    "phi_vec",
    "phitilde_vec",
]
