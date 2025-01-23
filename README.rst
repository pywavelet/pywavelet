pywavelet
#########

.. image:: https://badge.fury.io/py/pywavelet.svg
    :target: https://badge.fury.io/py/pywavelet
.. image:: https://coveralls.io/repos/github/avivajpeyi/pywavelet/badge.svg?branch=main&kill_cache=1
    :target: https://coveralls.io/github/avivajpeyi/pywavelet?branch=main





WDM Wavelet transform


Quickstart
==========

pywavelet is available on PyPI and can be installed with `pip <https://pip.pypa.io>`_.

.. code-block:: console

    $ pip install pywavelet

For developers
--------------

First set up a conda environment with the latest version of python.

.. code-block::

    $ conda create -n pywavelet -c conda-forge python=3.12

.. code-block::

    $ pip install -e ".[dev]"
    $ pre-commit install

Test code
---------

Locate directory /tests from root directory. run 

.. code-block::

    $ pytest .

Hopefully everything should run fine. 
