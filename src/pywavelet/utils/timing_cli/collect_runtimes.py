import time
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from ...backend import (
    cuda_available,
    get_dtype_from_env,
    get_precision_from_env,
)
from ...transforms.phi_computer import phitilde_vec_norm
from ...types import FrequencySeries

__all__ = ["collect_runtimes_for_backend"]


PRECISION = get_precision_from_env()
print(f"Using precision: {PRECISION}")
DTYPES = get_dtype_from_env()
print(f"Using dtypes: {DTYPES}")


def print_runtimes(runtimes: pd.DataFrame, label: str):
    # rich console to print the runtimes median +/- std

    console = Console()
    table = Table(title=f"Runtimes for {label}", box=box.SIMPLE)
    table.add_column("ND", justify="center")
    table.add_column("Runtime (s)", justify="center")

    for index, row in runtimes.iterrows():
        nd_log2 = int(np.log2(row["ND"]))
        table.add_row(
            f"2**{nd_log2}", f"{row['median']:.4f} +/- {row['std']:.4f}"
        )
    console.print(table)


def _generate_white_noise_freq_domain_dt(
    ND: int, dt: float = 0.0125, power_spectrum_level=1.0, seed=None
) -> FrequencySeries:
    """
    Generates a frequency domain representation of white noise given the
    time step (dt) and the length of the corresponding time-domain signal (ND).
    This function directly computes the positive frequency part of the
    frequency domain representation of white noise. The output length
    will be ND // 2 + 1.

    Args:
        dt (float): The time step between samples in the corresponding
                    time-domain signal (in seconds).
        ND (int): The length of the corresponding time-domain signal
                  (number of samples).
        power_spectrum_level (float, optional): The desired power spectral
                                               density level (per Hz).
                                               Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        numpy.ndarray: A complex-valued NumPy array representing the
                       positive frequency components of white noise.
                       The length will be ND // 2 + 1.
    """
    if seed is not None:
        np.random.seed(seed)

    n_positive_freqs = ND // 2 + 1
    df = 1.0 / (ND * dt)

    magnitudes = np.sqrt(power_spectrum_level * df) * np.ones(n_positive_freqs)
    phases = 2 * np.pi * np.random.rand(n_positive_freqs)
    positive_frequency_signal = magnitudes * np.exp(1j * phases)
    freqs = np.fft.rfftfreq(ND, dt)
    return FrequencySeries(positive_frequency_signal, freqs)


def _generate_func_args(ND: int, label="numpy") -> Tuple:
    Nf = Nt = int(np.sqrt(ND))
    yf = _generate_white_noise_freq_domain_dt(ND, seed=0).data
    phif = phitilde_vec_norm(Nf, Nt, d=4.0)

    if "jax" in label:
        from jax import numpy as jnp

        yf = jnp.array(yf, dtype=DTYPES[1])
        phif = jnp.array(phif, dtype=DTYPES[0])
    if "cupy" in label and cuda_available:
        import cupy as cp

        yf = cp.array(yf, dtype=DTYPES[1])
        phif = cp.array(phif, dtype=DTYPES[0])

    if "numpy" in label:
        yf = np.array(yf, dtype=DTYPES[1])
        phif = np.array(phif, dtype=DTYPES[0])

    return yf, Nf, Nt, phif, DTYPES[0], DTYPES[1]


def _collect_runtime(
    func: Callable, func_args: Tuple, nrep: int = 5
) -> Tuple[float, float]:
    # Single warm-up run to reduce startup overhead
    func(*func_args)

    times = []
    for _ in range(nrep):
        start = time.process_time()
        func(*func_args)
        end = time.process_time()
        times.append(end - start)

    return np.median(times), np.std(times)


def _collect_runtimes(
    func: Callable,
    label: str,
    max_log2f: int,
    nrep: int = 5,
    outdir: str = ".",
) -> pd.DataFrame:
    # Generate a list of NF values from 2^max_log2f to 2^2
    nf_values = np.array([2**i for i in range(2, max_log2f + 1)]).astype(
        int
    )[::-1]

    results = np.zeros((len(nf_values), 3))
    bar = tqdm(nf_values, desc=f"Running {label}")
    for i, Nf in enumerate(bar):
        ND = Nf * Nf
        bar.set_postfix(ND=f"2**{int(np.log2(ND))}")
        func_args = _generate_func_args(ND, label)
        try:
            _times = _collect_runtime(func, func_args, nrep)
        except Exception as e:
            print(f"Error processing ND={ND}: {e}")
            _times = (np.nan, np.nan)
        results[i] = np.array([ND, *_times])

    runtimes = pd.DataFrame(results, columns=["ND", "median", "std"])
    runtimes.to_csv(f"{outdir}/runtime_{label}.csv", index=False)

    print_runtimes(runtimes, label)
    return runtimes


def _collect_jax_runtimes(*args, **kwargs):
    import jax

    from pywavelet.transforms.jax.forward.from_freq import (
        transform_wavelet_freq_helper as jax_transform,
    )

    JAX_DEVICE = jax.default_backend()

    jax_label = f"jax_{JAX_DEVICE}_{PRECISION}"
    _collect_runtimes(jax_transform, jax_label, *args, **kwargs)


def _collect_cupy_runtimes(*args, **kwargs):
    from pywavelet.transforms.cupy.forward.from_freq import (
        transform_wavelet_freq_helper as cp_transform,
    )

    _collect_runtimes(cp_transform, f"cupy_{PRECISION}", *args, **kwargs)


def _collect_numpy_runtimes(*args, **kwargs):
    from pywavelet.transforms.numpy.forward.from_freq import (
        transform_wavelet_freq_helper as np_transform,
    )

    _collect_runtimes(np_transform, f"numpy_{PRECISION}", *args, **kwargs)


def collect_runtimes_for_backend(
    backend: str, max_log2f: int, nrep: int = 5, outdir: str = "."
):
    if "numpy" in backend:
        _collect_numpy_runtimes(max_log2f, nrep, outdir)
    elif "cupy" in backend:
        _collect_cupy_runtimes(max_log2f, nrep, outdir)
    elif "jax" in backend:
        _collect_jax_runtimes(max_log2f, nrep, outdir)
    else:
        raise ValueError(f"Unknown backend: {backend}")
