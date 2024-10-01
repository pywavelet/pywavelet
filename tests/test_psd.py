import numpy as np

from pywavelet.utils import evolutionary_psd_from_stationary_psd


def test_evolutionary_psd_from_stationary_psd(plot_dir):
    Nt, Nf = 64, 2048
    psd_f = np.linspace(0, 1024, 1024)
    psd = psd_f**-2
    t_grid = np.linspace(0, 1, Nt)
    f_grid = np.linspace(0, max(psd_f), Nf)

    ND = Nf * Nt
    dt = t_grid[-1] / ND

    w = evolutionary_psd_from_stationary_psd(
        psd=psd,
        psd_f=psd_f,
        f_grid=f_grid,
        t_grid=t_grid,
        dt=dt,
    )
    assert w.data.shape == (Nf, Nt)
