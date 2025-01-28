from typing import Callable, Tuple, Union

from ..backend import xp
from ..logger import logger


def _len_check(d):
    if not xp.log2(len(d)).is_integer():
        logger.warning(f"Data length {len(d)} is suggested to be a power of 2")


def is_documented_by(original: Callable):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def fmt_time(seconds: float, units=False) -> Union[str, Tuple[str, str]]:
    """Returns formatted time and units [ms, s, min, hr, day]"""
    t, u = "", ""
    if seconds < 1e-3:
        t, u = f"{seconds * 1e6:.2f}", "µs"
    elif seconds < 1:
        t, u = f"{seconds * 1e3:.2f}", "ms"
    elif seconds < 60:
        t, u = f"{seconds:.2f}", "s"
    elif seconds < 60 * 60:
        t, u = f"{seconds / 60:.2f}", "min"
    elif seconds < 60 * 60 * 24:
        t, u = f"{seconds / 3600:.2f}", "hr"
    else:
        t, u = f"{seconds / 86400:.2f}", "day"

    if units:
        return t, u
    return t


def fmt_timerange(trange):
    t0 = fmt_time(trange[0])
    tend, units = fmt_time(trange[1], units=True)
    return f"[{t0}, {tend}] {units}"


def fmt_pow2(n: float) -> str:
    pow2 = xp.log2(n)
    if pow2.is_integer():
        return f"2^{int(pow2)}"
    return f"{n:,}"
