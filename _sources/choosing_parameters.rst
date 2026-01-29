###################
Choosing parameters
###################

PyWavelet exposes a few key parameters that control the time–frequency grid and the
window overlap.


The hard constraint: input length
*********************************

For the time-domain transform, your input must have length:

.. math::

   ND = Nt \times Nf

If your data does not naturally have that length, pad or crop it first (see below).


How ``Nt`` and ``Nf`` affect resolution
**************************************

For a signal of duration ``T``:

.. math::

   \Delta T = T / Nt

.. math::

   \Delta F = \frac{1}{2\Delta T}

So increasing ``Nt`` gives finer time bins (smaller ``ΔT``) but makes each frequency bin
wider (larger ``ΔF``). The grid is a deliberate time–frequency tradeoff.


How ``mult`` affects overlap
****************************

``mult`` controls the window support length ``K = 2 * mult * Nf`` (see :doc:`window_function`).

Rules of thumb:

- start with ``mult = 16`` or ``32`` for exploratory plots
- if you see leakage, increase ``mult`` (up to roughly ``Nt/2``)
- if runtime is too slow, decrease ``mult`` before changing ``Nt``/``Nf``


Padding/cropping recipes
************************

If you want to keep a fixed ``dt`` and choose an ``Nt × Nf`` grid:

.. code-block:: python

   ND = Nt * Nf
   y = y[:ND]  # crop

Or pad:

.. code-block:: python

   import numpy as np
   ND = Nt * Nf
   if len(y) < ND:
       y = np.pad(y, (0, ND - len(y)))

If you just want a convenient length (power-of-two FFTs), consider the helper:

.. code-block:: python

   from pywavelet.types import TimeSeries
   ts = TimeSeries(data=y, time=t).zero_pad_to_power_of_2(tukey_window_alpha=0.1)


Backend/precision
*****************

If you're tuning performance, also see :doc:`backends` for switching NumPy/JAX/CuPy and
float32/float64.
