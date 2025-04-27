import os

import numpy as np
import pytest
from utils import cuda_available

from pywavelet.transforms.phi_computer import phitilde_vec_norm

# 8x8 monochromatic signal
WNM = np.array(
    [
        [
            -2.17231658e-16,
            -2.17231658e-16,
            1.82849722e-15,
            1.82849722e-15,
            -1.59693281e-15,
            -1.59693281e-15,
            -1.14095073e-15,
            -1.14095073e-15,
        ],
        [
            5.21849164e-15,
            -1.92598987e-15,
            1.69405022e-15,
            -6.11340860e-18,
            4.95368161e-16,
            6.95072219e-16,
            1.56088961e-14,
            -3.23455137e-15,
        ],
        [
            6.27820524e-15,
            -7.39243293e-16,
            -1.75734889e-15,
            -1.08780986e-14,
            3.19246930e-15,
            -1.41771970e-14,
            1.35153646e-15,
            -1.52447448e-14,
        ],
        [
            -1.35115223e-14,
            1.52017787e-15,
            4.31116053e-15,
            -3.99699681e-16,
            -8.63776618e-16,
            5.22364583e-16,
            -1.34704878e-14,
            -1.09012981e-15,
        ],
        [
            -1.92753121e-14,
            7.99999952e00,
            -6.09730187e-15,
            7.99999952e00,
            -1.89903889e-14,
            7.99999952e00,
            -2.70376169e-14,
            7.99999952e00,
        ],
        [
            1.35115223e-14,
            1.52017787e-15,
            -4.31116053e-15,
            -3.99699681e-16,
            8.63776618e-16,
            5.22364583e-16,
            1.34704878e-14,
            -1.09012981e-15,
        ],
        [
            6.27820524e-15,
            7.39243293e-16,
            -1.75734889e-15,
            1.08780986e-14,
            3.19246930e-15,
            1.41771970e-14,
            1.35153646e-15,
            1.52447448e-14,
        ],
        [
            -5.21849164e-15,
            -1.92598987e-15,
            -1.69405022e-15,
            -6.11340860e-18,
            -4.95368161e-16,
            6.95072219e-16,
            -1.56088961e-14,
            -3.23455137e-15,
        ],
    ]
)
# Define the parameters
fs = 32
dt = 1 / fs
f0 = 8.0
Nt, Nf = 2**3, 2**3
ND = Nt * Nf
A = 2.0

expected_y = np.fft.rfft(
    A * np.sin(2 * np.pi * f0 * np.arange(0, ND) * dt)
) * np.sqrt(2)
phif = phitilde_vec_norm(Nf, Nt, 4.0)


def get_trasform_funct(backend):
    if backend == "numpy":
        from pywavelet.transforms.numpy.inverse.to_freq import (
            inverse_wavelet_freq_helper_fast as inverse_wavelet_freq_helper,
        )
    elif backend == "jax":
        from pywavelet.transforms.jax.inverse.to_freq import (
            inverse_wavelet_freq_helper,
        )
    elif backend == "cupy":
        from pywavelet.transforms.cupy.inverse.to_freq import (
            inverse_wavelet_freq_helper,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return inverse_wavelet_freq_helper


@pytest.mark.parametrize("backend", ["numpy", "jax", "cupy"])
def test_inverse_transform(backend, plot_dir):
    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")

    transform_func = get_trasform_funct(backend)
    # Perform the inverse transform
    reconstructed_signal = transform_func(WNM, phif, Nf, Nt)

    # Check if the shape of the reconstructed signal is correct
    assert (
        reconstructed_signal.shape == expected_y.shape
    ), f"Shape mismatch for backend {backend}"
    # Check if the reconstructed signal is close to the expected signal
    assert np.allclose(
        reconstructed_signal, expected_y, atol=1e-5
    ), f"Inverse transform failed for backend {backend}"

    # plot the results (for visualization purposes)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(
        np.abs(expected_y) ** 2, label="Expected Signal", lw=2, color="black"
    )
    plt.plot(
        np.abs(reconstructed_signal) ** 2,
        label="Reconstructed Signal",
        alpha=0.5,
        marker="o",
    )
    plt.yscale("log")
    plt.title(f"Inverse Transform Comparison ({backend})")
    plt.savefig(os.path.join(plot_dir, f"inverse_transform_{backend}.png"))
