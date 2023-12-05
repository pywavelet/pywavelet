import bilby
import numpy as np

from typing import Tuple
from pywavelet.transforms.types import TimeSeries


DURATION = 8
SAMPLING_FREQUENCY = 512
DT = 1 / SAMPLING_FREQUENCY
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = 256


CBC_GENERATOR = bilby.gw.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLING_FREQUENCY,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=20.0,
        minimum_frequency=MINIMUM_FREQUENCY,
        maximum_frequency=MAXIMUM_FREQUENCY,
    ),
)

GW_PARMS = dict(
    mass_1=30,
    mass_2=30,  # 2 mass parameters
    a_1=0.1,
    a_2=0.1,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,  # 6 spin parameters
    ra=1.375,
    dec=-1.2108,
    luminosity_distance=2000.0,
    theta_jn=0.0,  # 7 extrinsic parameters
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
)


def get_ifo(t0=0.0, noise=True):
    ifos = bilby.gw.detector.InterferometerList(["H1"])  # design sensitivity
    if noise:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=SAMPLING_FREQUENCY,

            duration=DURATION,
            start_time=t0,
        )
    else:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=t0,
        )
    return ifos


def inject_signal_in_noise(mc, q=1, distance=1000.0, noise=True) -> Tuple[TimeSeries, float]:
    injection_parameters = GW_PARMS.copy()
    (
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
    ) = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        mc, q
    )
    injection_parameters["luminosity_distance"] = distance

    ifos = get_ifo(injection_parameters["geocent_time"] + 1.5, noise=noise)
    ifos.inject_signal(
        waveform_generator=CBC_GENERATOR, parameters=injection_parameters
    )
    ifo: bilby.gw.detector.Interferometer = ifos[0]

    snr = ifo.meta_data["optimal_SNR"]

    data = TimeSeries(ifo.strain_data.time_domain_strain, ifo.time_array)
    return data, np.abs(snr)