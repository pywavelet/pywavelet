#############
API
#############


The main functionality of `pywavelet` is to help transform/inverse transform data from the
time/frequency domains to the WDM-wavelet domain.


********************
Transform Functions
********************

.. autofunction:: pywavelet.transforms.from_time_to_wavelet
.. autofunction:: pywavelet.transforms.from_freq_to_wavelet
.. autofunction:: pywavelet.transforms.from_wavelet_to_freq
.. autofunction:: pywavelet.transforms.from_wavelet_to_time

**************************
Types
**************************

.. autoclass:: pywavelet.transforms.types.TimeSeries
    :members:
    :undoc-members:

.. autoclass:: pywavelet.transforms.types.FrequencySeries
    :members:
    :undoc-members:


.. autoclass:: pywavelet.transforms.types.Wavelet
    :members:
    :undoc-members:
