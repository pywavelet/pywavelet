PyWavelet
---------

`PyWavelet` is a Python package for transforming 1D data into the wavelet domain.

To install the package, run the following command:

.. code:: bash

   pip install pywavelet


.. raw:: html

   <img src="_static/demo.gif" alt="Demo GIF" loop="false" style="max-width: 80%; height: auto;">

   <hr style="margin: 1.25rem 0; border: 0; border-top: 1px solid #d0d7de; max-width: 80%;">

   <div style="display:flex; gap: 1.25rem; flex-wrap: wrap; align-items: flex-start; justify-content: left; margin-top: 1rem;">
     <figure style="margin: 0; max-width: 420px;">
       <figcaption style="font-weight: 600; margin-bottom: 0.35rem;">Time → wavelet</figcaption>
       <img src="_static/time_to_wavelet.gif" alt="Time to wavelet packetization animation" style="max-width: 420px; height: auto;">
     </figure>
     <figure style="margin: 0; max-width: 420px;">
       <figcaption style="font-weight: 600; margin-bottom: 0.35rem;">Frequency → wavelet</figcaption>
       <img src="_static/freq_to_wavelet.gif" alt="Frequency to wavelet packetization animation" style="max-width: 420px; height: auto;">
     </figure>
   </div>

   <div style="margin-top: 1rem;">
     <figure style="margin: 0; max-width: 420px;">
       <figcaption style="font-weight: 600; margin-bottom: 0.35rem;">Conceptual time–frequency track</figcaption>
       <img src="_static/chirp_animation.svg" alt="Conceptual time-frequency track animation" style="max-width: 420px; height: auto;">
     </figure>
   </div>

Where to start
**************

- :doc:`quickstart` for a minimal transform/inverse-transform example
- :doc:`how_it_works` for the conceptual overview
- :doc:`backends` for NumPy/JAX/CuPy and precision selection
- :doc:`choosing_parameters` for choosing ``Nf``/``Nt``/``mult``
- :doc:`window_function` for the window (``phi`` / ``phitilde``) explanation
- :doc:`api` for the full API reference
- the notebooks under ``examples/`` for worked examples and plots


Acknowledging PyWavelet
***********************

If you use `PyWavelet` in your research, please acknowledge it by citing the works in :doc:`citing`.
