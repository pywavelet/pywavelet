from typing import Union

import numpy as np

from ..transforms.types import FrequencySeries, TimeSeries


def _preprocess_bins(
    data: Union[TimeSeries, FrequencySeries], Nf=None, Nt=None
):
    """preprocess the bins"""

    if data.name == "Frequency Series":
        N = 2*len(data)
    elif data.name == "Time Series":
        N = len(data)

    if Nt is not None and Nf is None:
        assert 1 <= Nt <= N, f"Nt={Nt} must be between 1 and N={N}"
        Nf = N // Nt

    elif Nf is not None and Nt is None:
        assert 1 <= Nf <= N, f"Nf={Nf} must be between 1 and N={N}"
        Nt = N // Nf

    _N = Nf * Nt
    return Nf, Nt


def _get_bins(data: Union[TimeSeries, FrequencySeries], Nf=None, Nt=None):
    """Get the bins for the wavelet transform
    Eq 4-6 in Wavelets paper
    """
    T = data.duration
    if data.name == "Frequency Series":
        N = len(data)
    elif data.name == "Time Series":
        N = len(data)

    fs = N / T
    fmax = fs / 2

    delta_T = T / Nt
    delta_F = 1 / (2 * delta_T)

    # assert delta_f == fmax / Nf, f"delta_f={delta_f} != fmax/Nf={fmax/Nf}"

    f_bins = np.arange(0, Nf) * delta_F
    t_bins = np.arange(0, Nt) * delta_T

    if isinstance(data, TimeSeries):
        t_bins += data.time[0]

    return t_bins, f_bins
