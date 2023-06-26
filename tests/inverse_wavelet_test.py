""""test that both inverse functions perform as specified in stored dat files"""
from time import perf_counter
import numpy as np

import pytest

from pywavelet.transforms import from_wavelet_to_time, from_wavelet_to_freq, inverse_wavelet_freq_time

import pywavelet.fft_funcs as fft

#whether to expect exact match for input files
EXACT_MATCH = False

def test_inverse_wavelets():
    """test that inverse wavelet transforms perform precisely as recorded in the input dat files
    for random input data"""
    #transform parameters
    file_wave = 'rand_wavelet.dat'
    file_freq = 'rand_wave_freq.dat'
    file_time = 'rand_wave_time.dat'

    dt = 30.

    #get a wavelet representation of a signal
    print('begin loading data files')
    t0 = perf_counter()

    #the initial data
    wave_in = np.loadtxt(file_wave)

    #the precomputed inverse frequency domain transform
    fs_in,signal_freq_real_in,signal_freq_im_in = np.loadtxt(file_freq).T
    signal_freq_in = signal_freq_real_in+1j*signal_freq_im_in

    #the precomputed inverse time domain transform
    ts_in,signal_time_in = np.loadtxt(file_time).T
    t1 = perf_counter()
    print('loaded input files in %5.3fs'%(t1-t0))

    Nt = wave_in.shape[0]
    Nf = wave_in.shape[1]

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    ts = np.arange(0,ND)*dt
    fs = np.arange(0,ND//2+1)*1/(Tobs)

    assert np.all(ts_in==ts)
    assert np.all(fs_in==fs)


    signal_freq = from_wavelet_to_freq(wave_in, Nf, Nt)

    t0 = perf_counter()
    signal_freq = from_wavelet_to_freq(wave_in, Nf, Nt)
    t1 = perf_counter()

    print('got frequency domain transform in %5.3fs'%(t1-t0))



    t2 = perf_counter()
    signal_time_trans = fft.irfft(signal_freq)
    t3 = perf_counter()

    print('got inverse fourier transform in %5.3fs'%(t3-t2))


    signal_time = from_wavelet_to_time(wave_in, Nf, Nt, mult=32)

    t4 = perf_counter()
    signal_time = from_wavelet_to_time(wave_in, Nf, Nt, mult=32)
    t5 = perf_counter()

    print('got time domain transform in %5.3fs'%(t5-t4))


    t6 = perf_counter()
    signal_freq_trans = fft.rfft(signal_time)
    t7 = perf_counter()

    print('got forward fourier transform in %5.3fs'%(t7-t6))

    signal_time2 = inverse_wavelet_freq_time(wave_in,Nf,Nt)

    t8 = perf_counter()
    signal_time2 = inverse_wavelet_freq_time(wave_in,Nf,Nt)
    t9 = perf_counter()

    print('got inverse wavelet in time domain via fft in %5.3fs'%(t9-t8))


    #check the files themselves
    if EXACT_MATCH:
        assert np.all(signal_freq==signal_freq_in)
        print('input wavelet domain data matches input frequency domain data')

        assert np.all(signal_time==signal_time_in)
        print('input wavelet domain data matches input time domain data')
    else:
        assert np.allclose(signal_freq,signal_freq_in,atol=1.e-12,rtol=1.e-12)
        print('input wavelet domain data matches input frequency domain data')

        assert np.allclose(signal_time,signal_time_in,atol=1.e-14,rtol=1.e-15)
        print('input wavelet domain data matches input time domain data')

    #some additional internal consistency checkes
    assert np.allclose(signal_time_trans,signal_time,atol=1.e-4,rtol=1.e-10)
    print('expected level of internal consistency between time domain representations')

    assert np.allclose(np.real(signal_freq_trans),np.real(signal_freq),atol=1.e-2,rtol=1.e-2)
    assert np.allclose(np.imag(signal_freq_trans),np.imag(signal_freq),atol=1.e-2,rtol=1.e-2)
    assert np.allclose(signal_freq_trans,signal_freq,atol=1.e-2,rtol=1.e-2)
    print('expected level of internal consistency between freq domain representations')

    assert np.all(signal_time2 == signal_time_trans)
    print('shorthand function inverse_wavelet_freq_time matches')

    print('all tests passed')
