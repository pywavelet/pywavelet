import matplotlib.pyplot as plt
import numpy as np
from conftest import monochromatic_wnm

from pywavelet.types import Wavelet, WaveletMask
from pywavelet.utils import compute_likelihood


def test_mask(plot_dir):
    f0 = 20
    d = monochromatic_wnm(f0=f0)
    h = monochromatic_wnm(f0=f0)
    psd = Wavelet(np.ones((h.Nf, h.Nt)), h.time, h.freq)
    assert compute_likelihood(d, h, psd) == 0

    mask = WaveletMask.from_restrictions(h.time, h.freq, [f0 - 0.5, f0 + 0.5])
    dmasked = d * mask
    assert np.isclose(compute_likelihood(d, h, psd, mask), 0)
    assert np.isclose(compute_likelihood(dmasked, h, psd), 0)
    # number of nans in dmasked.data
    assert np.isnan(dmasked.data).sum() != 0

    mask1 = WaveletMask.from_restrictions(
        h.time, h.freq, [f0 + 0.5, f0 + 1.5], tgaps=[[1.7 * 60, 3.4 * 60]]
    )
    dmasked1 = d * mask1
    # assert np.isclose(compute_likelihood(d, h, psd_analysis, mask1), 0) == False

    # plt the 3 differnet datasets
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    d.plot(ax=axes[0])
    axes[0].set_title("data")
    (d * mask).plot(ax=axes[1])
    axes[1].set_title("data*mask[f0-0.5, f0+0.5]")
    (d * mask1).plot(ax=axes[2])
    axes[2].set_title("data*mask[f0+0.5, f0+1.5]")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/test_mask.png")
