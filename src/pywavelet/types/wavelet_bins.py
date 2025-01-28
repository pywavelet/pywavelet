from typing import Tuple, Union

from ..backend import xp
from .frequencyseries import FrequencySeries
from .timeseries import TimeSeries


def _preprocess_bins(
    data: Union[TimeSeries, FrequencySeries], Nf=None, Nt=None
) -> Tuple[int, int]:
    """preprocess the bins"""

    if isinstance(data, TimeSeries):
        N = len(data)
    elif isinstance(data, FrequencySeries):
        # len(d) =  N // 2 + 1
        N = 2 * (len(data) - 1)

    if Nt is not None and Nf is None:
        assert 1 <= Nt <= N, f"Nt={Nt} must be between 1 and N={N}"
        Nf = N // Nt

    elif Nf is not None and Nt is None:
        assert 1 <= Nf <= N, f"Nf={Nf} must be between 1 and N={N}"
        Nt = N // Nf

    _N = Nf * Nt
    return Nf, Nt


def _get_bins(
    data: Union[TimeSeries, FrequencySeries],
    Nf: Union[int, None] = None,
    Nt: Union[int, None] = None,
) -> Tuple[xp.ndarray, xp.ndarray]:
    T = data.duration
    t_bins, f_bins = compute_bins(Nf, Nt, T)

    # N = len(data)
    # fs = N / T
    # assert delta_f == fmax / Nf, f"delta_f={delta_f} != fmax/Nf={fmax/Nf}"

    t_bins += data.t0

    return t_bins, f_bins


def compute_bins(Nf: int, Nt: int, T: float) -> Tuple[xp.ndarray, xp.ndarray]:
    """Get the bins for the wavelet transform
    Eq 4-6 in Wavelets paper
    """
    delta_T = T / Nt
    delta_F = 1 / (2 * delta_T)
    t_bins = xp.arange(0, Nt) * delta_T
    f_bins = xp.arange(0, Nf) * delta_F
    return t_bins, f_bins
