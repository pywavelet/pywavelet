###########
Quickstart
###########

Install
*******

.. code-block:: bash

   pip install pywavelet


Minimal round-trip (time → wavelet → time)
******************************************

The WDM transform operates on a regular grid with:

- ``Nf`` frequency bins
- ``Nt`` time bins
- ``ND = Nf * Nt`` samples in the 1D input signal

.. code-block:: python

   import numpy as np
   from scipy.signal import chirp

   from pywavelet.types import TimeSeries
   from pywavelet.transforms import from_time_to_wavelet, from_wavelet_to_time

   dt = 1 / 512
   Nf = 64
   Nt = 64
   mult = 16
   ND = Nf * Nt

   t = np.arange(ND) * dt
   y = chirp(t, f0=10.0, f1=100.0, t1=t[-1], method="hyperbolic")

   h_time = TimeSeries(data=y, time=t)
   h_wavelet = from_time_to_wavelet(h_time, Nf=Nf, Nt=Nt, mult=mult)
   h_recon = from_wavelet_to_time(h_wavelet, dt=dt, mult=mult)

   # Quick visualization helpers
   _ = h_time.plot_spectrogram()
   _ = h_wavelet.plot(absolute=True, cmap="Reds")


Notes
*****

- If your input length is not exactly ``Nf * Nt``, pad/crop the data before calling the transform.
- The transform backend (NumPy/JAX/CuPy) is chosen at import time; see :doc:`backends` if you want GPU/JAX support.
