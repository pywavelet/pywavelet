import numpy as np

from ..logger import logger
from . import utils


class CoupledTimeAndFrequencySeries(object):
    def __init__(
        self,
        duration: float = None,
        sampling_frequency: float = None,
        start_time: float = 0,
        minimum_frequency: float = 0,
        maximum_frequency: float = np.inf,
        roll_off=0.2,
    ):
        self._duration: float = duration
        self._sampling_frequency: float = sampling_frequency
        self._start_time: float = start_time
        self._minimum_frequency: float = minimum_frequency
        self._maximum_frequency: float = maximum_frequency
        self._roll_off: float = roll_off
        self.window_factor = 1

        self._frequency_array_updated: bool = False
        self._time_array_updated: bool = False
        self._frequency_mask_updated: bool = False

        self._frequency_array: np.ndarray = None
        self._time_array: np.ndarray = None
        self._frequency_mask: np.ndarray = None
        self._frequency_domain_data: np.ndarray = None
        self._time_domain_data: np.ndarray = None

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(duration={self.duration}, sampling_frequency={self.sampling_frequency}, start_time={self.start_time})"
        )

    @property
    def alpha(self):
        return 2 * self.roll_off / self.duration

    def time_domain_window(self, roll_off=None, alpha=None):
        """
        Window function to apply to time domain data before FFTing.

        This defines self.window_factor as the power loss due to the windowing.
        See https://dcc.ligo.org/DocDB/0027/T040089/000/T040089-00.pdf

        Parameters
        ==========
        roll_off: float
            Rise time of window in seconds
        alpha: float
            Parameter to pass to tukey window, how much of segment falls
            into windowed part

        Returns
        =======
        window: array
            Window function over time array
        """
        from scipy.signal.windows import tukey

        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = alpha * self.duration / 2
        window = tukey(len(self._time_domain_strain), alpha=self.alpha)
        self.window_factor = np.mean(window**2)
        return window

    @property
    def frequency_array(self) -> np.ndarray:
        if not self._frequency_array_updated:
            if self.sampling_frequency and self.duration:
                self._frequency_array = utils.create_frequency_series(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration,
                )
            else:
                raise ValueError(
                    "Can not calculate a frequency series without a "
                    "legitimate sampling_frequency ({}) or duration ({})".format(
                        self.sampling_frequency, self.duration
                    )
                )

            self._frequency_array_updated = True
        return self._frequency_array

    @frequency_array.setter
    def frequency_array(self, frequency_array: np.ndarray) -> None:
        self._frequency_array = frequency_array
        (
            self._sampling_frequency,
            self._duration,
        ) = utils.get_sampling_frequency_and_duration_from_frequency_array(
            frequency_array
        )
        self._frequency_array_updated = True

    @property
    def time_array(self) -> np.ndarray:
        if not self._time_array_updated:
            if self.sampling_frequency and self.duration:
                self._time_array = utils.create_time_series(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration,
                    starting_time=self.start_time,
                )
            else:
                raise ValueError(
                    "Can not calculate a time series without a "
                    "legitimate sampling_frequency ({}) or duration ({})".format(
                        self.sampling_frequency, self.duration
                    )
                )

            self._time_array_updated = True
        return self._time_array

    @time_array.setter
    def time_array(self, time_array: np.ndarray) -> None:
        self._time_array = time_array
        (
            self._sampling_frequency,
            self._duration,
        ) = utils.get_sampling_frequency_and_duration_from_time_array(
            time_array
        )
        self._start_time = time_array[0]
        self._time_array_updated = True

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration
        self._frequency_array_updated = False
        self._time_array_updated = False

    @property
    def sampling_frequency(self) -> float:
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency: float) -> None:
        self._sampling_frequency = sampling_frequency
        self._frequency_array_updated = False
        self._time_array_updated = False

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: float) -> None:
        self._start_time = start_time
        self._time_array_updated = False

    @property
    def minimum_frequency(self) -> float:
        return self._minimum_frequency

    @minimum_frequency.setter
    def minimum_frequency(self, minimum_frequency: float) -> None:
        self._minimum_frequency = minimum_frequency
        self._frequency_mask_updated = False

    @property
    def maximum_frequency(self) -> float:
        if self.sampling_frequency is not None:
            if 2 * self._maximum_frequency > self.sampling_frequency:
                self._maximum_frequency = self.sampling_frequency / 2.0
        return self._maximum_frequency

    @maximum_frequency.setter
    def maximum_frequency(self, maximum_frequency: float) -> None:
        self._maximum_frequency = maximum_frequency
        self._frequency_mask_updated = False

    @property
    def frequency_mask(self) -> np.ndarray:
        if not self._frequency_mask_updated:
            frequency_array = self.frequency_array
            mask = (frequency_array >= self.minimum_frequency) & (
                frequency_array <= self.maximum_frequency
            )
            self._frequency_mask = mask
            self._frequency_mask_updated = True
        return self._frequency_mask

    @frequency_mask.setter
    def frequency_mask(self, mask: np.ndarray) -> None:
        self._frequency_mask = mask
        self._frequency_mask_updated = True

    @property
    def time_domain_data(self) -> np.ndarray:
        if self._time_domain_data is not None:
            return self._time_domain_data
        elif self._frequency_domain_data is not None:
            self._time_domain_data = utils.infft(
                self.frequency_domain_data, self.sampling_frequency
            )
            return self._time_domain_data
        else:
            raise ValueError("time domain strain data not yet set")

    @property
    def frequency_domain_data(self) -> np.ndarray:
        if self._frequency_domain_data is not None:
            return self._frequency_domain_data * self.frequency_mask
        elif self._time_domain_data is not None:
            logger.debug(
                "Generating frequency domain strain from given time "
                "domain strain."
            )
            logger.debug(
                "Applying a tukey window with alpha={}, roll off={}".format(
                    self.alpha, self.roll_off
                )
            )
            self.low_pass_filter()
            window = self.time_domain_window()
            self._frequency_domain_strain, self.frequency_array = utils.nfft(
                self._time_domain_strain * window, self.sampling_frequency
            )
            return self._frequency_domain_strain * self.frequency_mask

            self._frequency_domain_data, self.frequency_array = utils.nfft(
                self._time_domain_data, self.sampling_frequency
            )
            return self._frequency_domain_data * self.frequency_mask
        else:
            raise ValueError("frequency domain strain data not yet set")

    @frequency_domain_data.setter
    def frequency_domain_data(self, frequency_domain_data: np.ndarray) -> None:
        if not len(self.frequency_array) == len(frequency_domain_data):
            raise ValueError(
                "The frequency_array and the set strain have different lengths"
            )
        self._frequency_domain_data = frequency_domain_data
        self._time_domain_data = None

    def _infer_time_domain_dependence(
        self,
        start_time: float,
        sampling_frequency: float,
        duration: float,
        time_array: np.ndarray,
    ) -> None:
        self._infer_dependence(
            domain="time",
            array=time_array,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
        )

    def _infer_frequency_domain_dependence(
        self,
        start_time: float,
        sampling_frequency: float,
        duration: float,
        frequency_array: np.ndarray,
    ) -> None:
        self._infer_dependence(
            domain="frequency",
            array=frequency_array,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
        )

    def _infer_dependence(
        self,
        domain: str,
        array: np.ndarray,
        duration: float,
        sampling_frequency: float,
        start_time: float,
    ) -> None:
        if (sampling_frequency is not None) and (duration is not None):
            if array is not None:
                raise ValueError(
                    "You have given the sampling_frequency, duration, and "
                    "an array"
                )
            pass
        elif array is not None:
            if domain == "time":
                self.time_array = array
            elif domain == "frequency":
                self.frequency_array = array
                self.start_time = start_time
            return
        elif sampling_frequency is None or duration is None:
            raise ValueError(
                "You must provide both sampling_frequency and duration"
            )
        else:
            raise ValueError("Insufficient information given to set arrays")
        self._duration = duration
        self._sampling_frequency = sampling_frequency
        self._start_time = start_time

    @classmethod
    def set_from_time_domain(
        cls,
        time_domain_data: np.ndarray,
        sampling_frequency: float = None,
        duration: float = None,
        start_time: float = 0,
        time_array: np.ndarray = None,
    ) -> "CoupledTimeAndFrequencySeries":
        self = cls(
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
        )

        self._infer_time_domain_dependence(
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            duration=duration,
            time_array=time_array,
        )

        logger.debug("Setting data using provided time_domain_data")
        if np.shape(time_domain_data) == np.shape(self.time_array):
            self._time_domain_data = time_domain_data
            self._frequency_domain_data = None
        else:
            raise ValueError("Data times do not match time array")
        return self

    @classmethod
    def set_from_frequency_domain(
        cls,
        frequency_domain_data: np.ndarray,
        sampling_frequency: float = None,
        duration: float = None,
        start_time: float = 0,
        frequency_array: np.ndarray = None,
    ) -> "CoupledTimeAndFrequencySeries":
        self = cls(
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
        )

        self._infer_frequency_domain_dependence(
            start_time=start_time,
            sampling_frequency=sampling_frequency,
            duration=duration,
            frequency_array=frequency_array,
        )

        logger.debug("Setting data using provided frequency_domain_data")
        if np.shape(frequency_domain_data) == np.shape(self.frequency_array):
            self._frequency_domain_data = frequency_domain_data
        else:
            raise ValueError("Data frequencies do not match frequency_array")
        return self
