import numpy as np

from pywavelet.utils import evolutionary_psd_from_stationary_psd


def test_evolutionary_psd_from_stationary_psd(plot_dir):
    psd_f = np.linspace(0, 1024, 1024)
    psd = psd_f**-2
    t_grid = np.linspace(0, 1, 64)
    f_grid = np.linspace(0, max(psd_f), 2048)

    N_t = len(t_grid)
    N_f = len(f_grid)
    N = N_t * N_f
    dt = t_grid[-1] / N

    w = evolutionary_psd_from_stationary_psd(
        psd=psd,
        psd_f=psd_f,
        f_grid=f_grid,
        t_grid=t_grid,
        dt=dt,
    )
    assert w.data.shape == (N_f, N_t)
