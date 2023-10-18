from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from xarray_dataclasses import AsDataArray, Attr, Coord, Coordof, Data, Name

TIME = Literal["time"]
FREQ = Literal["freq"]


@dataclass
class TimeAxis:
    data: Data[TIME, int]
    long_name: Attr[str] = "Time"
    units: Attr[str] = "s"


@dataclass
class FreqAxis:
    data: Data[FREQ, int]
    long_name: Attr[str] = "Frequency"
    units: Attr[str] = "Hz"


@dataclass
class Wavelet(AsDataArray):
    data: Data[Tuple[TIME, FREQ], float]
    time: Coordof[TimeAxis] = 0
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Wavelet Amplitutde"


@dataclass
class TimeSeries(AsDataArray):
    data: Data[TIME, float]
    time: Coordof[TimeAxis] = 0
    name: Name[str] = "Time Series"


@dataclass
class FrequencySeries(AsDataArray):
    data: Data[FREQ, float]
    freq: Coordof[FreqAxis] = 0
    name: Name[str] = "Frequency Series"


def wavelet_dataset(
    wavelet_data: np.ndarray, time_grid=None, freq_grid=None, Nt=None, Nf=None
) -> Wavelet:
    """Create a dataset with wavelet coefficients.

    Parameters
    ----------
    wavelet : pywavelets.Wavelet object
        Wavelet to use.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with wavelet coefficients.
    """
    if Nt is None:
        Nt = wavelet_data.shape[1]
    if Nf is None:
        Nf = wavelet_data.shape[0]

    if time_grid is None:
        time_grid = np.arange(Nt)
    if freq_grid is None:
        freq_grid = np.arange(Nf)

    return Wavelet.new(wavelet_data, time=time_grid, freq=freq_grid)
