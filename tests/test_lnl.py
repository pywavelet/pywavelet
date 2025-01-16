import matplotlib.pyplot as plt
import numpy as np
from conftest import monochromatic_wnm

from pywavelet.types import Wavelet
from pywavelet.utils import compute_likelihood


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
