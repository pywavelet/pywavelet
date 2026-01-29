###
FAQ
###

Why doesn't changing the backend do anything?
*********************************************

The backend is selected **at import time**. Set ``PYWAVELET_BACKEND`` (and optionally
``PYWAVELET_PRECISION``) before importing ``pywavelet.transforms``. See :doc:`backends`.


Why is my reconstruction imperfect?
***********************************

Common causes:

- ``mult`` is too small (not enough overlap) → increase it (see :doc:`window_function`)
- you are cropping/padding inconsistently (time grid vs data length)
- you are mixing ``dt`` or durations between forward and inverse transforms


Why do I see NaNs in plots?
***************************

NaNs usually come from NaNs in the input data, or from downstream processing (e.g. whitening arrays).
Try plotting the input time/frequency series first and checking for invalid values.


How do I choose ``Nt`` and ``Nf``?
**********************************

Start from the constraint ``ND = Nt * Nf`` and decide whether you care more about:

- time localization → larger ``Nt``
- frequency localization → smaller ``Nt``

See :doc:`choosing_parameters` for the formulas and rules of thumb.


How do I cite PyWavelet?
************************

See :doc:`citing` and :file:`docs/references.bib`.

