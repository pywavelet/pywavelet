import importlib
import os

import numpy as np


def test_common_formatters_and_len_check(caplog):
    from pywavelet.types.common import _len_check, fmt_pow2, fmt_time

    # fmt_time
    assert fmt_time(1e-6, units=True)[1] == "µs"
    assert fmt_time(1e-2, units=True)[1] == "ms"
    assert fmt_time(1.0, units=True)[1] == "s"
    assert fmt_time(61.0, units=True)[1] == "min"
    assert fmt_time(3601.0, units=True)[1] == "hr"
    assert fmt_time(90000.0, units=True)[1] == "day"

    # fmt_pow2
    assert fmt_pow2(8) == "2^3"
    assert fmt_pow2(10) == "10"

    # _len_check warning path (non power-of-2 length)
    caplog.set_level("WARNING")
    _len_check(np.zeros(3))
    assert any("power of 2" in rec.message for rec in caplog.records)


def test_backend_invalid_backend_falls_back_to_numpy():
    import pywavelet.backend as backend

    old_backend = os.environ.get("PYWAVELET_BACKEND")
    old_precision = os.environ.get("PYWAVELET_PRECISION")
    try:
        os.environ["PYWAVELET_BACKEND"] = "not-a-backend"
        os.environ["PYWAVELET_PRECISION"] = "float32"
        importlib.reload(backend)
        assert backend.current_backend == "numpy"
        assert os.environ.get("PYWAVELET_BACKEND") == "numpy"
    finally:
        if old_backend is None:
            os.environ.pop("PYWAVELET_BACKEND", None)
        else:
            os.environ["PYWAVELET_BACKEND"] = old_backend
        if old_precision is None:
            os.environ.pop("PYWAVELET_PRECISION", None)
        else:
            os.environ["PYWAVELET_PRECISION"] = old_precision
        importlib.reload(backend)


def test_inverse_to_time_noncompact_helpers_and_wrap_branch():
    from pywavelet.transforms.numpy.inverse import to_time as tt

    Nf = 4
    Nt = 4
    ND = Nf * Nt

    # Cover the K+Nf > ND wrap branch in __core()
    mult = 3
    K = mult * 2 * Nf
    wave_in = np.arange(Nt * Nf, dtype=np.float64).reshape(Nt, Nf)
    phi = np.ones(K, dtype=np.float64)
    out = tt.inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult=mult)
    assert out.shape == (ND,)
    assert np.isfinite(out).all()

    # Cover non-compact pack/unpack helpers (not used by the main path)
    afins = np.zeros(2 * Nf, dtype=np.complex128)
    tt.pack_wave_time_helper(0, Nf, Nt, wave_in, afins)
    assert afins[0] == np.sqrt(2) * wave_in[0, 0]
    assert afins[Nf] == np.sqrt(2) * wave_in[1, 0]

    afins2 = np.zeros(2 * Nf, dtype=np.complex128)
    tt.pack_wave_time_helper(1, Nf, Nt, wave_in, afins2)
    assert afins2[0] == 0.0
    assert afins2[Nf] == 0.0

    K2 = 12
    phis = np.ones(K2, dtype=np.float64)
    fft_fin_real = np.arange(2 * Nf, dtype=np.float64)
    res = np.zeros(ND, dtype=np.float64)
    tt.unpack_time_wave_helper(
        n=0, Nf=Nf, Nt=Nt, K=K2, phis=phis, fft_fin_real=fft_fin_real, res=res
    )
    assert np.any(res != 0.0)
