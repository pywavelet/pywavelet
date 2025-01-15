import matplotlib.pyplot as plt
import numpy as np
from conftest import monochromatic_wnm

from pywavelet.transforms import (
    compute_bins,
    from_freq_to_wavelet,
    from_time_to_wavelet,
)
from pywavelet.transforms.types import (
    FrequencySeries,
    TimeSeries,
    Wavelet,
    WaveletMask,
)
from pywavelet.utils import (
    compute_likelihood,
    compute_snr,
    evolutionary_psd_from_stationary_psd,
)


def test_lnl(plot_dir):
    f0 = 20
    d = monochromatic_wnm(f0=f0)
    h = monochromatic_wnm(f0=f0)
    h2 = monochromatic_wnm(f0=f0 + 1)
    psd = Wavelet(np.ones((h.Nf, h.Nt)), h.time, h.freq)
    lnl = compute_likelihood(d, h, psd)
    lnl2 = compute_likelihood(d, h2, psd)
    assert lnl == 0
    assert lnl > lnl2
