import os

import numpy as np
import pytest
from utils import cuda_available
import matplotlib.pyplot as plt

from pywavelet.transforms.phi_computer import phitilde_vec_norm

fs = 32
dt = 1 / fs
f0 = 8.0
Nt, Nf = 2 ** 3, 2 ** 3
ND = Nt * Nf
A = 2.0

y_fft = np.fft.rfft(
    A * np.sin(2 * np.pi * f0 * np.arange(0, ND) * dt)
) * np.sqrt(2)
phif = phitilde_vec_norm(Nf, Nt, 4.0)

expected_wdm = np.array(
    [[-8.68926491e-16, -8.68926491e-16, 7.31398828e-15,
      7.31398828e-15, -6.38773173e-15, -6.38773173e-15,
      -4.56380304e-15, -4.56380304e-15],
     [2.08739652e-14, -7.70395919e-15, 6.77620216e-15,
      -2.44531235e-17, 1.98147152e-15, 2.78028760e-15,
      6.24355831e-14, -1.29382065e-14],
     [2.51128192e-14, -2.95697505e-15, -7.02939482e-15,
      -4.35123938e-14, 1.27698787e-14, -5.67087896e-14,
      5.40614582e-15, -6.09789829e-14],
     [-5.40460972e-14, 6.08071019e-15, 1.72446416e-14,
      -1.59879588e-15, -3.45510344e-15, 2.08945941e-15,
      -5.38819545e-14, -4.36052150e-15],
     [-7.71012509e-14, 3.20000000e+01, -2.43892118e-14,
      3.20000000e+01, -7.59615587e-14, 3.20000000e+01,
      -1.08150468e-13, 3.20000000e+01],
     [5.40460972e-14, 6.08071019e-15, -1.72446416e-14,
      -1.59879588e-15, 3.45510344e-15, 2.08945941e-15,
      5.38819545e-14, -4.36052150e-15],
     [2.51128192e-14, 2.95697505e-15, -7.02939482e-15,
      4.35123938e-14, 1.27698787e-14, 5.67087896e-14,
      5.40614582e-15, 6.09789829e-14],
     [-2.08739652e-14, -7.70395919e-15, -6.77620216e-15,
      -2.44531235e-17, -1.98147152e-15, 2.78028760e-15,
      -6.24355831e-14, -1.29382065e-14]
     ])


def get_trasform_funct(backend):
    if backend == "numpy":
        from pywavelet.transforms.numpy.forward.from_freq import (
            transform_wavelet_freq_helper
        )
    elif backend == "jax":
        from pywavelet.transforms.jax.forward.from_freq import (
            transform_wavelet_freq_helper,
        )
    elif backend == "cupy":
        from pywavelet.transforms.cupy.forward.from_freq import (
            transform_wavelet_freq_helper
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return transform_wavelet_freq_helper


@pytest.mark.parametrize("backend", ["numpy", "jax", "cupy"])
def test_forward_transform(backend, plot_dir):
    if backend == "cupy" and not cuda_available:
        pytest.skip("CUDA is not available")

    d = f"{plot_dir}/forward_transform"
    os.makedirs(d, exist_ok=True)

    transform_func = get_trasform_funct(backend)
    wdm = transform_func(
        y_fft,
        Nf,
        Nt,
        phif,
    )
    wdm_np = np.asarray(wdm).real
    diff = np.abs(wdm_np - expected_wdm)
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    cbar1 = fig.colorbar(axes[0].pcolormesh(wdm_np, shading="auto"), ax=axes[0])
    cbar2 = fig.colorbar(axes[1].pcolormesh(diff, shading="auto"), ax=axes[1])
    axes[0].set_title("Forward transform")
    axes[1].set_title("Diff")
    plt.tight_layout()
    plt.savefig(f"{d}/wdm_{backend}.png")
    np.testing.assert_allclose(expected_wdm, wdm_np.real)
