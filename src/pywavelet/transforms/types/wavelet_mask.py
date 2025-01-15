from typing import List

from .common import fmt_timerange, is_documented_by, xp


class WaveletMask:
    def __init__(
        self,
        mask: xp.ndarray,
        time: xp.ndarray,
        freq: xp.ndarray,
    ):
        self.mask = mask
        self.time = time
        self.freq = freq

    def __repr__(self):
        return f"WaveletMask({self.mask.shape}, {fmt_timerange(self.time)}, {self.freq})"

    @classmethod
    def from_grid(cls, time_grid: xp.ndarray, freq_grid: xp.ndarray):
        nt, nf = len(time_grid), len(freq_grid)
        mask = xp.zeros((nf, nt), dtype=bool)
        return cls(mask, time_grid, freq_grid)

    @classmethod
    def from_frange(
        cls, time_grid: xp.ndarray, freq_grid: xp.ndarray, frange: List[float]
    ):
        self = cls.from_grid(time_grid, freq_grid)
        self.mask[
            (freq_grid >= frange[0]) & (freq_grid <= frange[1]), :
        ] = True
        return self
