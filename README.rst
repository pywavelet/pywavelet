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


Note: We have transforms availible in numpy, JAX and Cupy.


For developers
--------------

First set up a conda environment with python 3.10

.. code-block::

    $ mamba create -n pywavelet python=3.10

.. code-block::

    $ CONDA_OVERRIDE_CUDA=12.4  mamba install "jaxlib=*=*cuda*" jax -c conda-forge
    $ CONDA_OVERRIDE_CUDA=12.4  conda install -c conda-forge cupy-core
    $ pip install -e ".[dev]"
    $ pre-commit install

Test code
---------

Locate directory /tests from root directory. run

.. code-block::

    $ pytest .

Hopefully everything should run fine.
