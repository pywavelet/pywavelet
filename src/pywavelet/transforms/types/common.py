from typing import Literal

import numpy as xp
from numpy.fft import irfft, fft, rfft, rfftfreq

from ...logger import logger



def _len_check(d):
    if not np.log2(len(d)).is_integer():
        logger.warning(f"Data length {len(d)} is suggested to be a power of 2")


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper
