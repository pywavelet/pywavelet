from abc import ABC, abstractmethod


class WaveformGenerator(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, **params):
        """Call the waveform generator (using the lookup table) with the given parameters."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"