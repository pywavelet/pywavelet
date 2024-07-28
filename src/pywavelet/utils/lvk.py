from typing import Tuple

import bilby
import numpy as np
from scipy.interpolate import interp1d

from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils.snr import compute_frequency_optimal_snr

DURATION = 8
SAMPLING_FREQUENCY = 16384
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
    geocent_time=0,
)


def _get_ifo(t0=0.0, noise=True):
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


def inject_signal_in_noise(
    mc,
    q=1,
    distance=1000.0,
    noise=True,
) -> Tuple[FrequencySeries, FrequencySeries, float]:
    injection_parameters = GW_PARMS.copy()
    (
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
    ) = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        mc, q
    )
    injection_parameters["luminosity_distance"] = distance

    ifos = _get_ifo(injection_parameters["geocent_time"] + 1.5, noise=noise)
    ifos.inject_signal(
        waveform_generator=CBC_GENERATOR, parameters=injection_parameters
    )
    ifo: bilby.gw.detector.Interferometer = ifos[0]

    # This code will not work with our formalism due to non-integer power of two
    # for the lengths. 

    # fmask = ifo.frequency_mask
    # freq = ifo.strain_data.frequency_array[fmask]
    # psd = ifo.power_spectral_density_array[fmask]
    # h_rfft = ifo.frequency_domain_strain[fmask]


    # a band aid solution. We need to fix this.  
    freq = ifo.strain_data.frequency_array
    psd = ifo.power_spectral_density_array
    h_rfft = ifo.frequency_domain_strain

    # Create a mask to identify non-inf values
    finite_mask = np.isfinite(psd) 
    # Find the maximum value among the non-inf values
    max_value = np.max(psd[finite_mask]) 
    # Replace inf values with this maximum value
    psd[~finite_mask] = max_value
    # Here I am essentially regularising the PSD. 
    
    psd = FrequencySeries(psd, freq)
    h_f_cont = FrequencySeries(h_rfft, freq)
    h_f_disc = FrequencySeries(h_rfft/h_f_cont.dt, freq)
    snr = compute_frequency_optimal_snr(h_f_cont.data, psd, DURATION)

    return h_f_disc, psd, np.abs(snr)


def get_lvk_psd():
    ifo: bilby.gw.detector.Interferometer = _get_ifo()[0]
    psd = ifo.power_spectral_density.psd_array
    psd_f = ifo.power_spectral_density.frequency_array
    return psd, psd_f


def get_lvk_psd_function():
    psd, psd_f = get_lvk_psd()
    return interp1d(psd_f, psd, bounds_error=False, fill_value=max(psd))
