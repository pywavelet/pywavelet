""""test that both inverse functions perform as specified in stored dat files"""
from time import perf_counter
import numpy as np

import pytest

from pywavelet.transforms import from_time_to_wavelet, from_freq_to_wavelet, from_time_to_freq_to_wavelet

import pywavelet.fft_funcs as fft
import matplotlib.pyplot as plt

EXACT_MATCH = False

def test_forward_wavelets():
    """test that forward wavelet transforms perform precisely as recorded in the input dat files
    for random input data"""
    file_freq = 'rand_wave_freq.dat'
    file_time = 'rand_wave_time.dat'

    file_wave = 'rand_wavelet.dat'
    file_wave_freq = 'rand_wavelet_freq.dat'
    file_wave_time = 'rand_wavelet_time.dat'

    dt = 30.

    #get a wavelet representation of a signal
    print('begin loading data files')
    t0 = perf_counter()
    #the original data (wave_in)
    wave_in = np.loadtxt(file_wave)
    plt.imshow(wave_in.T)
    plt.xlabel("time [s]")
    plt.ylabel("frequency [hz]")
    plt.savefig("wavein.png")
    plt.show()


    #the forward wavelet transform of wave_in inverse wavelet transformed using frequency domain transforms both ways
    wave_freq_in = np.loadtxt(file_wave_freq)

    #the forward wavelet transform of wave_in inverse wavelet transformed time frequency domain transforms both ways
    wave_time_in = np.loadtxt(file_wave_time)

    #frequency domain forward and inverse transform is almost lossless so these should be nearly identical
    assert np.allclose(wave_freq_in,wave_in,atol=1.e-14,rtol=1.e-15)

    #time domain forward and inverse transforms are not quite as accurate so these have a little more tolerance
    assert np.allclose(wave_in,wave_time_in,atol=1.e-6,rtol=1.e-6)
    assert np.allclose(wave_freq_in,wave_time_in,atol=1.e-6,rtol=1.e-6)

    #the frequency domain inverse wavelet transform of wave_in
    fs_in,signal_freq_real_in,signal_freq_im_in = np.loadtxt(file_freq).T

    #the time domain inverse wvaelet transform of wave_in
    signal_freq_in = signal_freq_real_in+1j*signal_freq_im_in
    plt.plot(fs_in, signal_freq_in)
    plt.savefig("wavein_f.png")

    ts_in,signal_time_in = np.loadtxt(file_time).T

    plt.close()
    plt.plot(ts_in, signal_time_in)
    plt.savefig("wavein_t.png")


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

    t0 = perf_counter()
    wave_freq_got = from_freq_to_wavelet(signal_freq_in, Nf, Nt)
    t1 = perf_counter()

    print('got frequency domain transform in %5.3fs'%(t1-t0))


    t0 = perf_counter()
    wave_time_got = from_time_to_wavelet(signal_time_in, Nf, Nt, mult=32)
    t1 = perf_counter()
    print('got time domain forward transform in %5.3fs'%(t1-t0))


    t0 = perf_counter()
    wave_time_got2 = from_time_to_freq_to_wavelet(signal_time_in, Nf, Nt)
    t1 = perf_counter()

    plt.plot(wave_time_got2)
    plt.savefig("test.png")

    print('got from time domain to wavelet domain via fft in %5.3fs'%(t1-t0))

    #needed for internal consistency check of wave_time_got2
    wave_time_got3 = from_freq_to_wavelet(fft.rfft(signal_time_in), Nf, Nt)

    if EXACT_MATCH:
        assert np.all(wave_freq_got==wave_freq_in)
        print('forward frequency domain transform matches expectation exactly')
        print(wave_time_got[0,0])
        print(wave_time_in[0,0])
        print(wave_freq_got[0,0])
        print(wave_time_got==wave_time_in)
        print(np.sum(wave_time_got==wave_time_in))
        assert np.all(wave_time_got==wave_time_in)
        print('forward time domain transform matches expectation exactly')
    else:
        #on different architecture than originally generated the files (i.e. different fft implementations)
        #match may not be exact but should still be close
        assert np.allclose(wave_freq_in,wave_freq_got,atol=1.e-14,rtol=1.e-15)
        print('forward frequency domain transform matches expectation closely')

        assert np.allclose(wave_time_in,wave_time_got,atol=1.e-14,rtol=1.e-15)
        print('forward time domain transform matches expectation closely')

    #still expect good match within the given architecture
    assert np.allclose(wave_freq_got,wave_time_got,atol=1.e-6,rtol=1.e-6)
    print('transforms match as closely as expected')

    #internal consistency of helper function
    assert np.all(wave_time_got3==wave_time_got2)

    print('all tests passed')
