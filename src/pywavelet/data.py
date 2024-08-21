# TODO: do we need to keep this class? if we do, lets nuke the bilby dependency (#https://github.com/pywavelet/pywavelet/issues/14)
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import PropertyAccessor
from bilby.gw.detector.strain_data import (
    InterferometerStrainData as _BilbyData,
)

from .logger import logger
from .transforms import (
    from_freq_to_wavelet,
    from_time_to_wavelet,
    from_wavelet_to_time,
)
from .transforms.types import FrequencySeries, TimeSeries, Wavelet


class CoupledData(object):
    """A class to hold data that is coupled in the time, frequency, wavelet domains"""

    duration = PropertyAccessor("_coupled_timefreq_data", "duration")
    sampling_frequency = PropertyAccessor(
        "_coupled_timefreq_data", "sampling_frequency"
    )
    start_time = PropertyAccessor("_coupled_timefreq_data", "start_time")
    _frequency_array = PropertyAccessor(
        "_coupled_timefreq_data", "frequency_array"
    )
    _time_array = PropertyAccessor("_coupled_timefreq_data", "time_array")
    minimum_frequency = PropertyAccessor(
        "_coupled_timefreq_data", "minimum_frequency"
    )
    maximum_frequency = PropertyAccessor(
        "_coupled_timefreq_data", "maximum_frequency"
    )

    def __init__(
        self,
        minimum_frequency=0,
        maximum_frequency=np.inf,
        roll_off=0.2,
        Nt=None,
        Nf=None,
        nx=4.0,
        mult=32,
    ):
        """Initiate a CoupledData object

        The initialised object contains no data, this should be added using one
        of the `set_from..` methods.

        Parameters
        ==========
        minimum_frequency: float
            Minimum frequency to analyse for detector. Default is 0.
        maximum_frequency: float
            Maximum frequency to analyse for detector. Default is infinity.
        roll_off: float
            The roll-off (in seconds) used in the Tukey window, default=0.2s.
            This corresponds to alpha * duration / 2 for scipy tukey window.

        """
        self._coupled_timefreq_data = _BilbyData(
            minimum_frequency, maximum_frequency, roll_off
        )
        # wavelet stuff
        self.Nf = Nf
        self.Nt = Nt
        self.nx = nx
        self.mult = mult
        self._wavelet = None

    @property
    def nyquist_frequency(self):
        return self.sampling_frequency / 2

    @property
    def timeseries(self) -> TimeSeries:
        return TimeSeries(
            self._coupled_timefreq_data.time_domain_strain, self._time_array
        )

    @property
    def frequencyseries(self) -> FrequencySeries:
        return FrequencySeries(
            self._coupled_timefreq_data.frequency_domain_strain,
            self._frequency_array,
        )

    @property
    def wavelet(self) -> Wavelet:
        return self._wavelet

    @classmethod
    def from_timeseries(
        cls,
        timeseries: TimeSeries,
        minimum_frequency=0,
        maximum_frequency=np.inf,
        roll_off=0.2,
        Nt=None,
        Nf=None,
        nx=4.0,
        mult=32,
    ):
        strain_data = cls(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            roll_off=roll_off,
            Nt=Nt,
            Nf=Nf,
            nx=nx,
            mult=mult,
        )
        strain_data._coupled_timefreq_data.set_from_time_domain_strain(
            timeseries.data,
            sampling_frequency=timeseries.fs,
            duration=timeseries.duration,
            start_time=timeseries.t0,
        )
        strain_data._wavelet = from_time_to_wavelet(
            data=timeseries,
            Nf=strain_data.Nf,
            Nt=strain_data.Nt,
            nx=strain_data.nx,
            mult=strain_data.mult,
        )
        return strain_data

    @classmethod
    def from_frequencyseries(
        cls,
        frequencyseries: FrequencySeries,
        start_time=0,
        roll_off=0.2,
        Nt=None,
        Nf=None,
        nx=4.0,
        mult=32,
    ):
        """
        freqseries: Single-sided FFT of time domain strain normalised to units of strain / Hz
        """
        min_f, max_f = frequencyseries.freq_range
        strain_data = cls(
            minimum_frequency=min_f,
            maximum_frequency=max_f,
            roll_off=roll_off,
            Nt=Nt,
            Nf=Nf,
            nx=nx,
            mult=mult,
        )
        strain_data._coupled_timefreq_data.set_from_frequency_domain_strain(
            frequencyseries.data,
            sampling_frequency=frequencyseries.fs,
            duration=frequencyseries.duration,
            start_time=start_time,
        )
        strain_data._wavelet = from_freq_to_wavelet(
            strain_data.frequencyseries,
            Nf=strain_data.Nf,
            Nt=strain_data.Nt,
            nx=strain_data.nx,
        )
        return strain_data

    @classmethod
    def from_wavelet(
        cls,
        wavelet: Wavelet,
        minimum_frequency=0,
        maximum_frequency=np.inf,
        roll_off=0.2,
        nx=4.0,
        mult=32.0,
        dt=None,
    ):
        strain_data = cls(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            roll_off=roll_off,
            Nt=wavelet.Nt,
            Nf=wavelet.Nf,
            nx=wavelet.nx,
            mult=wavelet.mult,
        )

        strain_data._wavelet = wavelet
        ts = from_wavelet_to_time(wavelet, nx=nx, mult=mult, dt=dt)
        strain_data._coupled_timefreq_data.set_from_time_domain_strain(ts)

        return strain_data

    def plot_periodogram(self, *args, **kwargs):
        fig, ax = self.frequencyseries.plot_periodogram(*args, **kwargs)
        ax.set_xlim(self.minimum_frequency, self.maximum_frequency)
        return fig, ax

    def plot_wavelet(self, *args, **kwargs):
        fig, ax = self.wavelet.plot(*args, **kwargs)
        ax.set_ylim(self.minimum_frequency, self.maximum_frequency)
        return fig, ax

    def plot_timeseries(self, *args, **kwargs):
        return self.timeseries.plot(*args, **kwargs)

    def plot_spectrogram(self, *args, **kwargs):
        fig, ax = self.timeseries.plot_spectrogram(*args, **kwargs)
        ax.set_ylim(self.minimum_frequency, self.maximum_frequency)
        return fig, ax

    def plot_all(
        self,
        axes=None,
        timeseries_kwgs={},
        periodogram_kwgs={},
        spectrogram_kwgs={},
        wavelet_kwgs={},
    ):
        if axes is None:
            fig, axes = plt.subplots(4, 1, figsize=(4, 14))
        fig = axes[0].get_figure()
        self.plot_timeseries(ax=axes[0], **timeseries_kwgs)
        self.plot_periodogram(ax=axes[1], **periodogram_kwgs)
        self.plot_spectrogram(ax=axes[2], **spectrogram_kwgs)
        self.plot_wavelet(ax=axes[3], **wavelet_kwgs)
        plt.tight_layout()
        return fig, axes

    @property
    def freq_range(self):
        return (self.minimum_frequency, self.maximum_frequency)


class Data(CoupledData):
    logger.warning(
        "The Data class is deprecated and will be removed in a future release. "
        "Please use CoupledData instead."
    )
