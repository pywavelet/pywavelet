from ..waveform_generator import WaveformGenerator
from ...transforms import from_time_to_wavelet


class LookupTableWaveformGenerator(WaveformGenerator):

    def __init__(self, name, func, Nf=1024, Nt=1024):
        super().__init__(name)
        self.func = func
        self.Nf = Nf
        self.Nt = Nt

    def __call__(self, **params):
        time_signal = self.func(**params)
        wavelet_signal = from_time_to_wavelet(time_signal, Nf=1024, Nt=1024)
        return wavelet_signal
