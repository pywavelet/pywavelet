import numpy as np
import pytest


def evolutionary_psd_from_stationary_psd(psd, psd_f, delta_f, f_range, Nt):
    """
    PSD[ti,fi] = PSD[fi] * delta_f
    """

    # now we interpolate the PSD and evaluate it on the grid
    f_grid = np.arange(f_range[0], f_range[1], delta_f)
    psd_grid = np.interp(f_grid, psd_f, psd)

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0)

    return psd_grid
