import numpy as np

from ...transforms import from_time_to_wavelet
from ...transforms.types import Wavelet
from ...utils.wavelet_bins import _preprocess_bins
from ..waveform_generator import WaveformGenerator


class FunctionalWaveformGenerator(WaveformGenerator):
    def __init__(self, func, Nf=None, Nt=None, mult=32):
        super().__init__("Functional")
        self.func = func
        # cant have non-none for both Nf and Nt
        assert (Nf is None) != (
            Nt is None
        ), "Must pass either Nf or Nt (not both)"

        self.Nf = Nf
        self.Nt = Nt
        self.mult = mult

    def __call__(self, **params) -> Wavelet:
        """
        Generate a waveform from a functional form.

        Parameters
        ----------
        params: dict
            A dictionary of parameters to pass to the functional form.

        Returns
        -------
        wavelet_signal: np.ndarray
            The waveform in the wavelet domain matrix of (Nt, Nf).
        """
        wavelet_signal = from_time_to_wavelet(
            self.func(**params), Nf=self.Nf, Nt=self.Nt, mult=self.mult
        )
        return wavelet_signal
