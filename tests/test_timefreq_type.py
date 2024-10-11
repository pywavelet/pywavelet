import numpy as np
from pywavelet.transforms.types import TimeSeries, FrequencySeries

def test_timefreq_type(sine_time):
    assert isinstance(sine_time, TimeSeries)
    sine_freq:FrequencySeries = sine_time.to_frequencyseries()
    assert len(sine_freq) == len(sine_time)//2 + 1
    assert sine_freq.duration == sine_time.duration
    assert sine_freq.minimum_frequency == 0
    assert sine_freq.fs == sine_time.fs

