####################
Backends & Precision
####################

PyWavelet supports multiple numerical backends:

- **NumPy** (default; always available)
- **JAX** (CPU/GPU/TPU, if installed)
- **CuPy** (CUDA GPU, if installed and CUDA is available)


Selecting a backend
*******************

The backend is selected via environment variables:

- ``PYWAVELET_BACKEND``: ``numpy`` | ``jax`` | ``cupy``
- ``PYWAVELET_PRECISION``: ``float32`` | ``float64``

Because the backend is chosen **at import time**, set these variables *before* importing
``pywavelet.transforms`` (or anything that imports it).

.. code-block:: bash

   PYWAVELET_BACKEND=jax PYWAVELET_PRECISION=float64 python -c \
     "from pywavelet.transforms import from_time_to_wavelet; print(from_time_to_wavelet)"

Or inside Python, before importing the transforms:

.. code-block:: python

   import os
   os.environ["PYWAVELET_BACKEND"] = "numpy"   # or "jax" / "cupy"
   os.environ["PYWAVELET_PRECISION"] = "float32"

   from pywavelet.transforms import from_time_to_wavelet


Installing optional backends
****************************

.. code-block:: bash

   pip install "pywavelet[jax]"
   pip install "pywavelet[cupy]"


Troubleshooting
***************

- If you request ``cupy`` without a working CUDA runtime, PyWavelet falls back to NumPy and prints a table of available backends.
- On JAX, ``PYWAVELET_PRECISION=float64`` enables 64-bit mode (``jax_enable_x64``) during import.
