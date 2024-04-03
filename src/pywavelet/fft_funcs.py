"""helper to make sure available fft functions are consistent across modules depending on install
mkl-fft is faster so it is the default, but numpy fft is probably more commonly installed to it is the fallback"""
# try:
#     import mkl_fft
#
#     rfft = mkl_fft.rfft_numpy
#     irfft = mkl_fft.irfft_numpy
#     fft = mkl_fft.fft
#     ifft = mkl_fft.ifft
#     fftfreq = mkl_fft.fftfreq
#
# except ImportError:
#     import numpy

import numpy
rfft = numpy.fft.rfft
irfft = numpy.fft.irfft
fft = numpy.fft.fft
ifft = numpy.fft.ifft
fftfreq = numpy.fft.fftfreq

