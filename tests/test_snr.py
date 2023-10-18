import bilby
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors

from pywavelet.transforms import from_time_to_wavelet
from pywavelet.utils.snr import compute_snr

Nf, Nt = 64, 64
mult = 16

DURATION = 8
SAMPLING_FREQUENCY = 512
MINIMUM_FREQUENCY = 20

CBC_GENERATOR = bilby.gw.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLING_FREQUENCY,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=20.0,
        minimum_frequency=MINIMUM_FREQUENCY,
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


def get_ifo(t0):
    ifos = bilby.gw.detector.InterferometerList(["H1"])  # design sensitivity
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=DURATION,
        start_time=t0,
    )
    return ifos


def inject_signal_in_noise(mc, q=1, distance=1000.0):
    injection_parameters = GW_PARMS.copy()
    (
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
    ) = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        mc, q
    )
    injection_parameters["luminosity_distance"] = distance

    ifos = get_ifo(injection_parameters["geocent_time"] + 1.5)
    ifos.inject_signal(
        waveform_generator=CBC_GENERATOR, parameters=injection_parameters
    )
    ifo: bilby.gw.detector.Interferometer = ifos[0]

    snr = ifo.meta_data["optimal_SNR"]
    return ifo.time_array, ifo.strain_data.time_domain_strain, np.abs(snr)


def get_noise_wavelet_data(t0):
    noise = get_ifo(t0)[0].strain_data.time_domain_strain
    noise_wavelet = from_time_to_wavelet(noise, Nf, Nt)
    return noise_wavelet


def get_wavelet_psd_from_median_noise(n=32):
    """n: number of noise wavelets to take median of"""
    noise_wavelets = []
    for i in range(n):
        np.random.seed(i)
        noise_wavelets.append(get_noise_wavelet_data(i * DURATION))
    return np.median(np.array(noise_wavelets), axis=0)


@pytest.skip("Not implemented")
def test_snr():
    t, h, _ = inject_signal_in_noise(mc=30, q=1, distance=0.1)
    _, data, time_domain_snr = inject_signal_in_noise(
        mc=30, q=1, distance=5000
    )
    data_wavelet = from_time_to_wavelet(data, Nf, Nt)
    h_wavelet = from_time_to_wavelet(h, Nf, Nt)
    psd_wavelet = get_wavelet_psd_from_median_noise()
    snr = compute_snr(h_wavelet, data_wavelet, psd_wavelet)
    assert isinstance(snr, float)
    assert snr > 10
