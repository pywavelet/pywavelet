import numpy as np
from pywavelet.types import FrequencySeries, TimeSeries
import matplotlib.pyplot as plt
from pywavelet.transforms.numpy import from_freq_to_wavelet

f0 = 20.0
dt = 0.0125
A = 2
Nt = 128
Nf = 256
ND = Nt * Nf
t = np.arange(0, ND) * dt

true_wdm_amp = A * np.sqrt(2 * Nf)

y = A * np.sin(2 * np.pi * f0 * t)
ts = TimeSeries(data=y, time=t)
fs = ts.to_frequencyseries()
numpy_wdm = from_freq_to_wavelet(fs, Nf=Nf)

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ts.plot(ax=ax[0])
fs.plot_periodogram(ax=ax[1])
numpy_wdm.plot(ax=ax[2])
plt.show()

print(ts.data.max())
print(fs.data.max())
print(numpy_wdm.data.max())
print(true_wdm_amp)

np.testing.assert_allclose(numpy_wdm.data.max(), true_wdm_amp, atol=1e-2)
