# Import every package under the sun.

# Created ~ 23:00 on 8th September, 2020.

import numpy as np

from LISA_utils import FFT, freq_PSD, inner_prod, waveform
from mcmc_fun import MCMC_run
import matplotlib.pyplot as plt
from corner import corner
np.random.seed(1234)

# Set true parameters. These are the parameters we want to estimate using MCMC.

a_true = 5e-21
f_true = 1e-3
fdot_true = 1e-8          

tmax =  120*60*60                 # Final time
fs = 2*f_true                     # Sampling rate
delta_t = np.floor(0.01/fs)       # Sampling interval -- largely oversampling here. 

t = np.arange(0,tmax,delta_t)     # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N_t = int(2**(np.ceil(np.log2(len(t)))))   # Round length of time series to a power of two. 
                                           # Length of time series

h_true_f = FFT(waveform(a_true,f_true,fdot_true,t))         # Compute true signal in
                                                            # frequency domain. Real signal so only considering
                                                            # positive frequencies here.

freq,PSD = freq_PSD(t,delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(h_true_f,h_true_f,PSD,delta_t,N_t)    # Compute optimal matched filtering SNR
print("SNR of source",np.sqrt(SNR2))
variance_noise_f = N_t * PSD / (4 * delta_t)            # Calculate variance of noise, real and imaginary.
N_f = len(variance_noise_f)                             # Length of signal in frequency domain
np.random.seed(1235)                                    # Set the seed

# Generate frequency domain noise
noise_f = np.random.normal(0,np.sqrt(variance_noise_f),N_f) + 1j*np.random.normal(0,np.sqrt(variance_noise_f),N_f) 

data_f = h_true_f + noise_f         # Construct data stream

# MCMC - parameter estimation

Ntotal = 30000  # Total number of iterations
burnin = 6000   # Set burn-in. This is the amount of samples we will discard whilst looking 
             # for the true parameters

variance_noise_f = N_t * PSD / (4 * delta_t)

delta_a = np.sqrt(7.8152977583191198e-46)
delta_f = np.sqrt(3.122370011848878e-17)
delta_dotf = np.sqrt(1.007508992696005e-27)

param_start = [a_true + 1000*delta_a, f_true + 750*delta_f, fdot_true - 750*delta_dotf]  # Starting values
true_vals = [a_true,f_true, fdot_true]   # True values

a_chain,f_chain,fdot_chain,lp  = MCMC_run(data_f, t, variance_noise_f,
                            Ntotal, param_start,
                            printerval = 5000, 
                            a_var_prop = delta_a**2,
                            f_var_prop = delta_f**2,
                            fdot_var_prop = delta_dotf**2)

print("Now printing summary statistics:")
print("Posterior mean value is E(a) = {0}, and standard deviation delta_a = {1}".format(np.mean(a_chain[burnin:]),np.sqrt(np.var(a_chain[burnin:]))))
print("Posterior mean value is E(f) = {0}, and standard deviation delta_f = {1}".format(np.mean(f_chain[burnin:]), np.sqrt(np.var(f_chain[burnin:]))))
print("Posterior mean value is E(fdot) = {0}, and standard deviation is delta_fdot = {1}".format(np.mean(fdot_chain),np.sqrt(np.var(fdot_chain[burnin:]))))


params = [r'$\log_{10}(a)$', r'$\log_{10}(f_{0})$', r'$\log_{10}(\dot{f}_{0})$']  # Set parameter labels
N_params = len(params)   # Set number of parameters to investigate
true_vals_for_plot = [np.log10(true_vals[0]),np.log10(true_vals[1]), np.log10(true_vals[2])]  # Set true values (log)

# Plot trace plot

a_chain_log = np.log10(a_chain)
f_chain_log = np.log10(np.array(f_chain))
fdot_chain_log = np.log10(np.array(fdot_chain))
samples = [a_chain_log, f_chain_log, fdot_chain_log]  # Store samples in a list
color = ['green','black','purple']  # Set pretty colours

fig,ax = plt.subplots(3,1)
for k in range(0,3):
    ax[k].plot(samples[k], color = color[k],label = "Accepted points")
    ax[k].axhline(y = true_vals_for_plot[k],c = 'red', linestyle='dashed', label = 'True value', )
    ax[k].set_xlabel('Iteration',fontsize = 10)
    ax[k].set_ylabel(params[k], fontsize = 10)
    ax[k].legend(loc = "upper right", fontsize = 12)
ax[0].set_title("Trace plots")
plt.show()
plt.clf()

# Plot trace plot after burnin

a_chain_log_burnin = a_chain_log[burnin:]   # Discard the first 0, ..., burnin samples from each chain
f_chain_log_burnin = f_chain_log[burnin:]
fdot_chain_log_burnin = fdot_chain_log[burnin:]
samples_burned = [a_chain_log_burnin, f_chain_log_burnin, fdot_chain_log_burnin]

fig,ax = plt.subplots(3,1)
for k in range(0,3):
    ax[k].plot(samples_burned[k], color = color[k],label = "Accepted points")
    ax[k].axhline(y = true_vals_for_plot[k],c = 'red', linestyle='dashed', label = 'True value')
    ax[k].set_xlabel('Iteration',fontsize = 10)
    ax[k].set_ylabel(params[k], fontsize = 10)
    ax[k].legend(loc = "upper right", fontsize = 12)
ax[0].set_title("Trace plots after burnin")
plt.show()
plt.clf()

# Plot corner plot

samples = np.column_stack(samples_burned)  # Stack samples to plot corner plot
figure = corner(samples,bins = 30, color = 'blue',plot_datapoints=False,smooth1d=True,
                    labels=params, 
                    label_kwargs = {"fontsize":12},set_xlabel = {'fontsize': 20},
                    show_titles=True, title_fmt='.7f',title_kwargs={"fontsize": 9},smooth = True)

axes = np.array(figure.axes).reshape((N_params, N_params))

for k in range(N_params):   # Plot true values on corner plot (diagonals)
    ax = axes[k, k]
    ax.axvline(true_vals_for_plot[k], color="r")  
        
for yi in range(N_params):  # Plot true values on corner plot (non-diagonals)
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axhline(true_vals_for_plot[yi], color="r")
        ax.axvline(true_vals_for_plot[xi],color= "r")
        ax.plot(true_vals_for_plot[xi], true_vals_for_plot[yi], "sr")
            
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=8)   # Set font of labels
plt.show()
plt.clf()