import matplotlib.pyplot as plt
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.waveform import get_td_waveform

from pywavelet.transforms.types import TimeAxis, TimeSeries


def cbc_waveform(mc, q=1, delta_t=1.0 / 4096, f_lower=20):
    m1 = mass1_from_mchirp_q(mc, q)
    m2 = mass2_from_mchirp_q(mc, q)
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        delta_t=delta_t,
        f_lower=f_lower,
    )
    data = TimeSeries(hp.data, time=TimeAxis(hp.sample_times.data))
    return data
