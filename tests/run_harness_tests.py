"""verify all the harness functions match expected results"""
import os
import numpy as np
import pytest


EXACT_MATCH = False

def test_harness_writes():
    """test that the command line harnesses match expected results"""
    filename_wavelet_in = 'rand_wavelet.dat'
    filename_freq_in = 'rand_wave_freq.dat'
    filename_time_in = 'rand_wave_time.dat'

    filename_fw_in = 'rand_wavelet_freq.dat'
    filename_tw_in = 'rand_wavelet_time.dat'

    filename_fw_out = 'rand_wavelet_freq_test.dat'
    filename_tw_out = 'rand_wavelet_time_test.dat'
    filename_tfw_out = 'rand_wavelet_time_freq_test.dat'
    filename_wf_out = 'rand_wave_freq_test.dat'
    filename_wt_out = 'rand_wave_time_test.dat'
    filename_wft_out = 'rand_wave_freq_time_test.dat'

    mult = 32
    dt = 30.
    Nt = 128
    Nf = 512

    os.system("python forward_wavelet_freq_harness.py "+filename_freq_in+" "+filename_fw_out+" "+str(dt)+" "+str(Nt)+" "+str(Nf))
    os.system("python forward_wavelet_time_harness.py "+filename_time_in+" "+filename_tw_out+" "+str(dt)+" "+str(Nt)+" "+str(Nf)+" "+str(mult))
    os.system("python forward_wavelet_freq_time_harness.py "+filename_time_in+" "+filename_tfw_out+" "+str(dt)+" "+str(Nt)+" "+str(Nf))
    os.system("python inverse_wavelet_freq_harness.py "+filename_wavelet_in+" "+filename_wf_out+" "+str(dt))
    os.system("python inverse_wavelet_time_harness.py "+filename_wavelet_in+" "+filename_wt_out+" "+str(dt)+" "+str(mult))
    os.system("python inverse_wavelet_freq_time_harness.py "+filename_wavelet_in+" "+filename_wft_out+" "+str(dt))

    w_in = np.loadtxt(filename_wavelet_in)
    fs_in,f_inr,f_ini = np.loadtxt(filename_freq_in).T
    f_in = f_inr+1j*f_ini
    ts_in,t_in = np.loadtxt(filename_time_in).T
    fw_in = np.loadtxt(filename_fw_in)
    tw_in = np.loadtxt(filename_tw_in)
    fw_out = np.loadtxt(filename_fw_out)
    tw_out = np.loadtxt(filename_tw_out)
    tfw_out = np.loadtxt(filename_tfw_out)
    fs_out,f_outr,f_outi = np.loadtxt(filename_wf_out).T
    f_out = f_outr+1j*f_ini
    ts_out1,t_out1 = np.loadtxt(filename_wt_out).T
    ts_out2,t_out2 = np.loadtxt(filename_wft_out).T

    #inherently exact tests
    assert np.all(ts_out1==ts_in)
    assert np.all(ts_out2==ts_in)
    assert np.all(fs_out==fs_in)

    #inherently approximate tests
    assert np.allclose(tfw_out,fw_in,atol=1.e-4,rtol=1.e-6)
    assert np.allclose(tfw_out,tw_in,atol=1.e-4,rtol=1.e-6)
    assert np.allclose(tw_out,fw_in,atol=1.e-6,rtol=1.e-6)

    assert np.allclose(tfw_out,w_in,atol=1.e-4,rtol=1.e-6)
    assert np.allclose(fw_out,w_in,atol=1.e-14,rtol=1.e-14)
    assert np.allclose(tw_out,w_in,atol=1.e-6,rtol=1.e-6)

    assert np.allclose(t_out1,t_out2,atol=1.e-4,rtol=1.e-6)
    assert np.allclose(t_in,t_out2,atol=1.e-4,rtol=1.e-6)
    if EXACT_MATCH:
        assert np.all(f_inr==f_outr)
        assert np.all(f_ini==f_outi)
        assert np.all(f_in==f_out)
        assert np.all(t_out1==t_in)
        assert np.all(fw_in==fw_out)
        assert np.all(tw_in==tw_out)
    else:
        assert np.allclose(f_inr,f_outr,atol=1.e-12,rtol=1.e-12)
        assert np.allclose(f_ini,f_outi,atol=1.e-12,rtol=1.e-12)
        assert np.allclose(f_in,f_out,atol=1.e-12,rtol=1.e-12)
        assert np.allclose(t_out1,t_in,atol=1.e-13,rtol=1.e-13)
        assert np.allclose(fw_in,fw_out,atol=1.e-13,rtol=1.e-13)
        assert np.allclose(tw_in,tw_out,atol=1.e-13,rtol=1.e-13)

    os.remove(filename_fw_out)
    os.remove(filename_tw_out)
    os.remove(filename_tfw_out)
    os.remove(filename_wf_out)
    os.remove(filename_wt_out)
    os.remove(filename_wft_out)

if __name__=='__main__':
    pytest.cmdline.main(['run_harness_tests.py'])
