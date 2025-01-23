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

    mask = WaveletMask.from_frange(h.time, h.freq, [f0 - 0.5, f0 + 0.5])
    assert np.isclose(compute_likelihood(d, h, psd, mask), 0)

    mask1 = WaveletMask.from_frange(h.time, h.freq, [f0 + 0.5, f0 + 1.5])
    # assert np.isclose(compute_likelihood(d, h, psd_analysis, mask1), 0) == False

    # plt the 3 differnet datasets
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    kwgs = dict(
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=[d.time[0], d.time[-1], d.freq[0], d.freq[-1]],
    )
    axes[0].imshow(d.data, **kwgs)
    axes[0].set_title("data")
    axes[1].imshow(d.data * mask.mask, **kwgs)
    axes[1].set_title("data*mask[f0-0.5, f0+0.5]")
    axes[2].imshow(d.data * mask1.mask, **kwgs)
    axes[2].set_title("data*mask[f0+0.5, f0+1.5]")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/test_mask.png")
