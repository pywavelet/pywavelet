import logging
import os

import click

VALID_BACKENDS = [
    "numpy32",
    "cupy32",
    "jax32",
    "jax64",
    "numpy64",
    "cupy64",
]


@click.command("pywavelet_timer")
@click.option(
    "-n",
    "--log2n",
    type=int,
    default=12,
    help="The maximum log2 of the number of frequencies.",
)
@click.option(
    "-r",
    "--nrep",
    type=int,
    default=5,
    help="The number of repetitions for each run.",
)
@click.option(
    "-o",
    "--outdir",
    type=str,
    default=".",
    help="The output directory for the CSV files.",
)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(VALID_BACKENDS),
    default="numpy",
    help="The backend to use for the computation.",
)
def cli_collect_runtime(
    log2n: int,
    nrep: int = 5,
    outdir: str = ".",
    backend: str = "numpy",
):
    """Collect runtimes for the specified mode and save to CSV files.

    Parameters
    ----------
    max_log2f : int
        The maximum log2 of the number of frequencies.
    nrep : int, optional
        The number of repetitions for each run, by default 5.
    outdir : str, optional
        The output directory for the CSV files, by default ".".
    backend : str, optional
        The backend to use for the computation, by default "numpy".
        Valid options are "numpy", "cupy", and "jax", 'jax64'.

    """
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend '{backend}'. Valid options are {VALID_BACKENDS}."
        )

    precision = "float64" if "64" in backend else "float32"

    if "64" in backend:
        import jax

        jax.config.update("jax_enable_x64", True)
    backend = backend[:-2]

    os.environ["PYWAVELET_BACKEND"] = backend
    os.environ["PYWAVELET_PRECISION"] = precision

    # Set up logging
    from pywavelet.logger import logger

    logger.setLevel(logging.ERROR)

    from pywavelet.utils.timing_cli.collect_runtimes import (
        collect_runtimes_for_backend,
    )
    from pywavelet.utils.timing_cli.plot import plot_runtimes

    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    collect_runtimes_for_backend(backend, log2n, nrep, outdir)
    plot_runtimes(outdir)
