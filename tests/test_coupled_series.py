import numpy as np
import pytest

from pywavelet.data.coupled_time_and_frequency_series import (
    CoupledTimeAndFrequencySeries,
)
from pywavelet.data.utils import create_frequency_series, create_time_series


def test_initialization():
    obj = CoupledTimeAndFrequencySeries()
    assert obj.duration is None
    assert obj.sampling_frequency is None
    assert obj.start_time == 0
    assert obj.minimum_frequency == 0
    assert obj.maximum_frequency == np.inf


def test_custom_initialization():
    obj = CoupledTimeAndFrequencySeries(
        duration=10.0,
        sampling_frequency=100.0,
        start_time=1.0,
        minimum_frequency=0.1,
        maximum_frequency=50.0,
    )
    assert obj.duration == 10.0
    assert obj.sampling_frequency == 100.0
    assert obj.start_time == 1.0
    assert obj.minimum_frequency == 0.1
    assert obj.maximum_frequency == 50.0


def test_frequency_array_setter_getter():
    obj = CoupledTimeAndFrequencySeries(
        sampling_frequency=100.0, duration=10.0
    )
    freq_array = np.linspace(0, 50, 100)
    obj.frequency_array = freq_array
    np.testing.assert_array_equal(obj.frequency_array, freq_array)


def test_time_array_setter_getter():
    obj = CoupledTimeAndFrequencySeries(
        sampling_frequency=100.0, duration=10.0
    )
    time_array = np.linspace(0, 10, 100)
    obj.time_array = time_array
    np.testing.assert_array_equal(obj.time_array, time_array)


def test_duration_setter_getter():
    obj = CoupledTimeAndFrequencySeries()
    obj.duration = 20.0
    assert obj.duration == 20.0


def test_sampling_frequency_setter_getter():
    obj = CoupledTimeAndFrequencySeries()
    obj.sampling_frequency = 200.0
    assert obj.sampling_frequency == 200.0


def test_start_time_setter_getter():
    obj = CoupledTimeAndFrequencySeries()
    obj.start_time = 2.0
    assert obj.start_time == 2.0


def test_minimum_frequency_setter_getter():
    obj = CoupledTimeAndFrequencySeries()
    obj.minimum_frequency = 0.5
    assert obj.minimum_frequency == 0.5


def test_maximum_frequency_setter_getter():
    obj = CoupledTimeAndFrequencySeries(sampling_frequency=100.0)
    obj.maximum_frequency = 60.0
    assert (
        obj.maximum_frequency == 50.0
    )  # Should be limited by Nyquist frequency


def test_frequency_mask_setter_getter():
    obj = CoupledTimeAndFrequencySeries(
        sampling_frequency=100.0, duration=10.0
    )
    freq_array = np.linspace(0, 50, 100)
    obj.frequency_array = freq_array
    mask = (freq_array >= 0.1) & (freq_array <= 50.0)
    obj.frequency_mask = mask
    np.testing.assert_array_equal(obj.frequency_mask, mask)


def test_set_from_time_domain():
    sampling_freq = 100
    duration = 10
    times = create_time_series(sampling_freq, duration)
    data = np.random.random(len(times))
    obj = CoupledTimeAndFrequencySeries.set_from_time_domain(
        data, sampling_frequency=sampling_freq, duration=duration
    )
    np.testing.assert_array_equal(obj.time_domain_data, data)


def test_set_from_frequency_domain():
    sampling_freq = 100
    duration = 10
    freq = create_frequency_series(sampling_freq, duration)
    data = np.random.random(len(freq)) + 1j * np.random.random(len(freq))

    obj = CoupledTimeAndFrequencySeries.set_from_frequency_domain(
        data, sampling_frequency=sampling_freq, duration=duration
    )
    np.testing.assert_array_equal(obj.frequency_domain_data, data)
