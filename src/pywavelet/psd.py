import bilby
import numpy as np
from scipy.interpolate import interp1d

from .transforms import from_time_to_wavelet
from .transforms.types import (
    FrequencySeries,
    TimeSeries,
    Wavelet,
    wavelet_dataset,
)


def evolutionary_psd_from_stationary_psd(
    psd: np.ndarray,
    psd_f: np.ndarray,
    f_grid,
    t_grid,
    Nt: int = None,
    Nf: int = None,
) -> Wavelet:
    """
    PSD[ti,fi] = PSD[fi] * delta_f
    """

    if Nt is None:
        Nt = len(t_grid)
    if Nf is None:
        Nf = len(f_grid)

    delta_f = f_grid[1] - f_grid[0]
    nan_val = np.nan
    psd_grid = (
        interp1d(
            psd_f, psd, kind="nearest", fill_value=nan_val, bounds_error=False
        )(f_grid)
        * delta_f
    )

    # repeat the PSD for each time bin
    psd_grid = np.repeat(psd_grid[None, :], Nt, axis=0)

    return wavelet_dataset(psd_grid, time_grid=t_grid, freq_grid=f_grid)


def _generate_noise_from_psd(
    psd, psd_f, duration, sampling_freq
) -> TimeSeries:
    psd_interp = interp1d(psd_f, psd, bounds_error=False, fill_value=np.inf)
    white_noise, frequencies = bilby.utils.create_white_noise(
        sampling_freq, duration
    )
    with np.errstate(invalid="ignore"):
        frequency_domain_strain = psd_interp(frequencies) ** 0.5 * white_noise
    out_of_bounds = (frequencies < min(psd_f)) | (frequencies > max(psd_f))
    frequency_domain_strain[out_of_bounds] = 0 * (1 + 1j)
    frq_series = FrequencySeries(
        data=frequency_domain_strain, freq=frequencies
    )
    return TimeSeries.from_frequency_series(frq_series)


def get_noise_wavelet_from_psd(
    duration, sampling_freq, psd_f, psd, Nf=None, Nt=None
) -> Wavelet:
    psd_noise_ts = _generate_noise_from_psd(
        psd, psd_f, duration, sampling_freq
    )
    psd_noise_wavelet = from_time_to_wavelet(psd_noise_ts, Nf=Nf, Nt=Nt)
    # nan out any values for freq outside of the range of the psd_freq
    nan_mask = (psd_noise_wavelet.freq.data < min(psd_f)) | (
        psd_noise_wavelet.freq.data > max(psd_f)
    )
    psd_noise_wavelet.data[nan_mask] = np.nan
    return psd_noise_wavelet
