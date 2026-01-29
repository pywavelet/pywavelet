##############
Window effects
##############

This page shows how the window parameters change:

- the window itself (``phi`` and ``phitilde``), and
- the resulting time–frequency representation (and reconstruction quality).

If you're new to the terminology, start with :doc:`window_function`.

Original signal
***************

All comparisons below use the same simple chirp-like input.

.. image:: _static/window_effects_original_signal.png
   :alt: Original signal and spectrogram used for window comparisons
   :width: 100%


Packetization animation
************************

Each time bin corresponds to a windowed slice of the spectrum that is inverse-FFT'd and then
packed into a column of the wavelet grid.

.. raw:: html

   <img src="_static/time_to_wavelet.gif" alt="Time to wavelet packetization animation" style="max-width: 100%; height: auto;">

Frequency → wavelet intuition
*****************************

If you start from a frequency-domain series (``from_freq_to_wavelet``), you can think of it
as sliding a frequency-domain window (``phitilde``), inverse-FFT'ing that packet, and then
packing it into the wavelet grid.

.. raw:: html

   <img src="_static/freq_to_wavelet.gif" alt="Frequency to wavelet packetization animation" style="max-width: 100%; height: auto;">


What ``nx`` changes (shape of the taper)
****************************************

.. image:: _static/phi_sweep.png
   :alt: phi(t) for different nx
   :width: 90%

.. image:: _static/phitilde_sweep.png
   :alt: phitilde(omega) for different nx
   :width: 90%


What ``mult`` changes (overlap/support)
***************************************

``mult`` increases the window support length ``K = 2 * mult * Nf``. Larger values usually
reduce leakage and improve reconstruction, at the cost of runtime.

.. image:: _static/window_effects_residuals.png
   :alt: Residual time series for different nx/mult
   :width: 100%

.. image:: _static/window_effects_wavelet.png
   :alt: Wavelet magnitude and residuals for different nx/mult
   :width: 100%
