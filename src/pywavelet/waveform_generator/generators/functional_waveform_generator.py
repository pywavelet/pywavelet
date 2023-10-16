from ..waveform_generator import WaveformGenerator
from ...transforms import from_time_to_wavelet

import numpy as np

class FunctionalWaveformGenerator(WaveformGenerator):

    def __init__(self,  func,Nf=1024, Nt=1024, mult=32):
        super().__init__("Functional")
        self.func = func
        self.Nf = Nf
        self.Nt = Nt
        self.mult = mult

    def __call__(self, **params)->np.ndarray:
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
        ht = self.func(**params)
        wavelet_signal = from_time_to_wavelet(ht, Nf=self.Nf, Nt=self.Nt, mult=self.mult)
        return wavelet_signal
