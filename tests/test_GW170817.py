def __test_gw170817():
    import os

    import gwpy
    import matplotlib.pyplot as plt
    import numpy as np
    import requests
    from gwpy.timeseries import TimeSeries as gwpyTimeSeries

    from pywavelet.psd import evolutionary_psd_from_stationary_psd
    from pywavelet.transforms.to_wavelets import from_time_to_wavelet
    from pywavelet.transforms.types import TimeSeries, Wavelet

    PSD_LINK = (
        "https://dcc.ligo.org/public/0158/P1900011/001/GWTC1_GW170817_PSDs.dat"
    )
    PSD_FNAME = "GWTC1_GW170817_PSDs.dat"
    # download the PSD file by downloading it from the link above
    # and placing it in the same directory as this script
    if not os.path.exists(PSD_FNAME):
        response = requests.get(PSD_LINK)
        with open(PSD_FNAME, "wb") as f:
            f.write(response.content)

    # Read the PSD file using numpy
    psd = np.loadtxt(PSD_FNAME)
    # col 0 = frequency
    # col 2 = L1 strain PSD
    psd_freq = psd[:, 0]
    psd_amp = psd[:, 2]

    # Set the GPS start and end times for GW170817
    gps_start = (
        1187008882.43 - 8
    )  # Example start time (replace with the actual start time)
    gps_end = (
        1187008882.43 + 2
    )  # Example end time (replace with the actual end time)

    # Download strain data
    strain = gwpyTimeSeries.fetch_open_data("L1", gps_start, gps_end)
    strain = TimeSeries(strain.value, time=strain.times.value)
    data_wavelet: Wavelet = from_time_to_wavelet(
        strain, Nf=1024, nx=4.0, mult=4
    )
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd_amp,
        psd_f=psd_freq,
        f_grid=data_wavelet.freq.data,
        t_grid=data_wavelet.time.data,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.log(psd_wavelet.data), aspect="auto")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_title("log PSD")
    axes[1].imshow(data_wavelet.data, aspect="auto")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("CoupledData Wavelet Amplitude")
    axes[2].imshow(data_wavelet.data / psd_wavelet.data, aspect="auto")
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_xlabel("Time [s]")
    plt.show()
