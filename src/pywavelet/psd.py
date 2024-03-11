import numpy as np
from scipy.interpolate import interp1d
from .transforms.types import wavelet_dataset, Wavelet
from typing import Tuple


def evolutionary_psd_from_stationary_psd(psd: np.ndarray, psd_f: np.ndarray,
                                         f_grid, t_grid, Nt: int=None) -> Wavelet:
    """
    PSD[ti,fi] = PSD[fi] * delta_f
    """

    Nt = len(t_grid) if Nt is None else Nt

    delta_f = f_grid[1] - f_grid[0]
    psd_grid = interp1d(psd_f, psd, kind='nearest', fill_value=np.nan, bounds_error=False)(f_grid) * delta_f

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0)

    return wavelet_dataset(psd_grid, time_grid=t_grid, freq_grid=f_grid)

