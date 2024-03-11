from typing import Union

import numpy as np

from ..transforms.types import FrequencySeries, TimeSeries


def _preprocess_bins(
        data: Union[TimeSeries, FrequencySeries], Nf=None, Nt=None
):
    """preprocess the bins"""

    # Can either pass Nf or Nt (not both)
    assert (Nf is None) != (Nt is None), "Must pass either Nf or Nt (not both)"

    N = len(data)

    # If Nt is passed, compute Nf
    if Nt is not None:
        assert 1 <= Nt <= N, f"Nt={Nt} must be between 1 and N={N}"
        Nf = N // Nt

    if Nf is not None:
        assert 1 <= Nf <= N, f"Nf={Nf} must be between 1 and N={N}"
        Nt = N // Nf

    return Nf, Nt


def _get_bins(data: Union[TimeSeries, FrequencySeries], Nf=None, Nt=None):
    if isinstance(data, TimeSeries):
        n = len(data)
        t_binwidth = Nf * data.dt
        f_binwidth = 1 / (2 * data.dt * Nf)

        assert t_binwidth * f_binwidth == 0.5, "t_binwidth * f_binwidth must be 0.5"
        fmax = 1 / (2 * data.dt)

        # generate bins based on binwidth Nt and Nf times
        t_bins = np.arange(0, data.duration, t_binwidth) + data.time[0]
        f_bins = np.arange(0, fmax, f_binwidth)

        assert len(t_bins) == Nt, f"len(t_bins)={len(t_bins)} must be Nt={Nt}"
        assert len(f_bins) == Nf, f"len(f_bins)={len(f_bins)} must be Nf={Nf}"
        # assert np.diff(t_bins)[0] * np.diff(f_bins)[0] == 0.5, "t_binwidth * f_binwidth must be 0.5"

    elif isinstance(data, FrequencySeries):
        raise NotImplementedError
    else:
        raise ValueError(f"Data type {type(data)} not recognized")

    return t_bins, f_bins
