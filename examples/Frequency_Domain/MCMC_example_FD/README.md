# Tutorials
This code uses a Markov-Chain Monte-Carlo algorithms in order to sample parameters of a toy gravitational wave signal buried in coloured LISA like noise. The waveform model we will use is

$$ h(t;a,f,\dot{f}) = a \sin \left(2\pi t \left[f + \frac{1}{2}\dot{f}t\right]\right) $$

and we aim to estimate the parameter set $\boldsymbol{\theta} = \{a,f,\dot{f}\}$ using various samplers

## Getting started
1. Install Anaconda if you do not have it.
2. Create a virtual environment and install numpy, scipy, matplotlib, corner, tqdm, jupyter


## The code structure -- metropolis
1. The script `LISA_utils.py` is a utility script containing useful python functions. Such functions include an approximate parametric model for the LISA power spectral density, for example.
2. The script `mcmc_func.py` includes the waveform model and scripts used to build the Metropolis sampler.
3. The script `mcmc.py` executes the metropolis algorithm. 

### How to use the code 

To execute the code:
1. Run `python mcmc.py`, requires dependencies from `LISA_utils.py` and `mcmc_fun.py`
2. Once the code has finished executing, it will plot the traceplots, cornerplots and print the summary statistics



