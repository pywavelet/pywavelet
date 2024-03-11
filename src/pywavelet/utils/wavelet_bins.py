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
    # TODO: This was the old version
    # t_binwidth = data.duration / Nt
    # f_binwidth = 1 / 2 * t_binwidth
    # fmax = 1 / (2 * data.dt)
    #
    # t_bins = np.linspace(data.time[0], data.time[-1], Nt)
    # f_bins = np.linspace(0, fmax, Nf)

    D = data.duration

    # QUENTIN'S VERSION:
    df = Nt / (2 * D)
    dt = D / Nt

    f_bins = np.arange(0, Nf) * df
    f_bins[0] = f_bins[1]  # avoid division by zero
    t_bins = np.arange(0, Nt) * dt

    if isinstance(data, TimeSeries):
        t_bins += data.time[0]

    return t_bins, f_bins
