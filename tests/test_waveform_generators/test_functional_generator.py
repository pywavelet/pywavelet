# import matplotlib.pyplot as plt
# import numpy as np
# from cbc_waveform import cbc_waveform
# from scipy.signal import spectrogram
#
#
#
#
# def test_waveform_function(plot_dir):
#     fig = plt.figure()
#     for i, mc in enumerate(range(10, 50, 5)):
#         ht = cbc_waveform(mc)
#         ht.data = ht.data / ht.data.max()
#         plt.plot(ht.time.data, ht.data + i, label=f"mc={mc}")
#     plt.legend()
#     plt.savefig(f"{plot_dir}/cbc_waveforms.png", dpi=300)
#
#
# def test_functional_generator(plot_dir):
#     dt = 1 / 256
#     fmin = 20
#     h_func = lambda mc: cbc_waveform(mc, q=1, delta_t=dt, f_lower=fmin)
#     Nt = 64
#     mult = 16
#     waveform_generator = FunctionalWaveformGenerator(h_func, Nt=Nt, mult=mult)
#
#     # time and frequency grids
#
#     for i, mc in enumerate(range(15, 50, 5)):
#         wavelet_matrix = waveform_generator(mc=mc)
#         print("I:", i, "mc:", mc, "wavelet_matrix:", wavelet_matrix.shape)
#         fig = wavelet_matrix.plot()
#         plt.suptitle(f"mc={mc}")
#         plt.savefig(f"{plot_dir}/wavelet_domain_{mc}.png", dpi=300)
#         plt.close("all")
#
#         # test spectogram
#         ht = cbc_waveform(mc)
#         t = ht.time.data
#         T = max(t)
#         fs = 1 / (t[1] - t[0])
#         ff, tt, Sxx = spectrogram(ht.data, fs=fs, nperseg=256, nfft=576)
#         plt.pcolormesh(tt, ff, Sxx)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Frequency (Hz)")
#         plt.ylim(0, 128)
#         plt.savefig(f"{plot_dir}/spectrogram_{mc}.png", dpi=300)
