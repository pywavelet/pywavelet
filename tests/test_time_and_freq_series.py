import matplotlib.pyplot as plt
import numpy as np
import pytest
import jax.numpy as jnp
from jax import random

from pywavelet.transforms.types import TimeSeries, FrequencySeries

key = random.PRNGKey(0)

# Constants
N = 1000
F0 = 30  # Frequency of the signal
DURATION = 10  # Duration of the signal in seconds

# Derived Constants
DT = DURATION / (N - 1)  # Time step
FS = 1 / DT  # Sampling frequency
NYQUIST = FS / 2  # Nyquist frequency
DF = 1 / DURATION  # Frequency resolution
FMAX = NYQUIST  # Maximum frequency (Nyquist)
N_POSITIVE_F = N // 2 + 1  # Number of positive frequencies


@pytest.fixture
def sample_time_series():
    """Generate a sample time series with a sinusoidal signal."""
    time = jnp.linspace(0, DURATION, N, endpoint=False)  # Time axis
    data = jnp.sin(2 * jnp.pi * F0 * time)  # Sinusoidal data
    return TimeSeries(data, time)  # Assuming TimeSeries class accepts (data, time)


@pytest.fixture
def sample_freq_series():
    """Generate a sample frequency series with a pure frequency at F0."""
    freq = jnp.linspace(0, FMAX, N_POSITIVE_F)  # Frequency axis for FFT
    data = jnp.zeros_like(freq)
    # Find the index where freq is closest to F0
    f0_idx = jnp.argmin(jnp.abs(freq - F0))
    data = data.at[f0_idx].set(1.0)  # Set it to 1.0 or any other magnitude you prefer
    return FrequencySeries(data, freq)  # Assuming FrequencySeries

def test_timeseries(sample_time_series, plot_dir):
    ts = sample_time_series

    # Test initialization and basic properties
    assert len(ts) == N
    assert ts.t0 == 0
    assert jnp.isclose(ts.tend, DURATION)
    assert jnp.isclose(ts.duration, DURATION)
    assert jnp.isclose(ts.fs, FS)
    assert jnp.isclose(ts.dt, DT)
    assert jnp.isclose(ts.nyquist_frequency, NYQUIST)

    # Test signal content
    time = jnp.linspace(0, DURATION, N)
    expected_data = jnp.sin(2 * jnp.pi * F0 * time)
    assert jnp.allclose(ts.data, expected_data)

    # Test arithmetic operations
    ts2 = TimeSeries(2 * ts.data, ts.time)
    assert jnp.allclose((ts + ts2).data, 3 * ts.data)
    assert jnp.allclose((ts2 - ts).data, ts.data)
    assert jnp.allclose((ts * 2).data, 2 * ts.data)
    assert jnp.allclose((ts2 / 2).data, ts.data)

    # Test conversion to frequency domain
    fs = ts.to_frequencyseries()
    assert isinstance(fs, FrequencySeries)
    assert len(fs) == FS
    assert jnp.isclose(fs.maximum_frequency, ts.nyquist_frequency)
    peak_freq = fs.freq[jnp.argmax(jnp.abs(fs.data))]
    assert jnp.isclose(peak_freq, F0, atol=fs.df)

    # Test plotting
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    ts.plot(ax=axes[0])
    ts.plot_spectrogram(ax=axes[1])
    plt.tight_layout()
    plt.savefig(plot_dir / "timeseries.png")



def test_freqseries(sample_freq_series, plot_dir):
    fs = sample_freq_series

    # Test initialization and basic properties
    assert len(fs) == N_POSITIVE_F
    assert fs.minimum_frequency == 0
    assert jnp.isclose(fs.maximum_frequency, FMAX)
    assert jnp.isclose(fs.df, FMAX / (N // 2))
    assert jnp.isclose(fs.nyquist_frequency, 50)

    # Test signal content
    freq = jnp.linspace(0, FMAX, N_POSITIVE_F)
    expected_data = jnp.exp(-((freq - F0) / 2) ** 2)
    assert jnp.allclose(fs.data, expected_data)

    # Test arithmetic operations
    fs2 = FrequencySeries(2 * fs.data, fs.freq)
    assert jnp.allclose((fs + fs2).data, 3 * fs.data)
    assert jnp.allclose((fs2 - fs).data, fs.data)
    assert jnp.allclose((fs * 2).data, 2 * fs.data)
    assert jnp.allclose((fs2 / 2).data, fs.data)

    # Test conversion to time domain
    ts = fs.to_timeseries()
    assert isinstance(ts, TimeSeries)
    assert len(ts) == N
    assert jnp.isclose(ts.nyquist_frequency, fs.maximum_frequency)
    fft_freq = jnp.fft.fftfreq(len(ts), d=ts.dt)
    fft_data = jnp.fft.fft(ts.data)
    peak_freq = fft_freq[jnp.argmax(jnp.abs(fft_data))]
    assert jnp.isclose(peak_freq, F0, atol=ts.fs / len(ts))

    # Test plotting
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    fs.plot(ax=axes[0])
    fs.plot_periodogram(ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/freqseries.png")


