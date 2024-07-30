from dataclasses import dataclass
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import spectrogram
from xarray_dataclasses import (
    AsDataArray,
    Attr,
    Coordof,
    Data,
    DataOptions,
    Name,
)

from ...logger import logger

TIME = Literal["time"]
FREQ = Literal["freq"]


@dataclass
class TimeAxis:
    data: Data[TIME, float]
    long_name: Attr[str] = "Time"
    units: Attr[str] = "s"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


@dataclass
class FreqAxis:
    data: Data[FREQ, float]
    long_name: Attr[str] = "Frequency"
    units: Attr[str] = "Hz"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def _len_check(d):
    if not np.log2(len(d)).is_integer():
        logger.warning(f"Data length {len(d)} is suggested to be a power of 2")
