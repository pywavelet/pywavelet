import pywavelet.transforms.numpy.forward.main as transformer
import importlib
from conftest import DATA_DIR, Nf, Nt, dt

import numpy as np
from pywavelet import set_backend


def test_np_precision(plot_dir, sine_freq):

    precisions = ["float32"]
    for precision in precisions:
        set_backend("numpy", precision)
        importlib.reload(transformer)

        w = transformer.from_freq_to_wavelet(
            sine_freq,
            Nf=Nf,
            Nt=Nt,
        )
        assert isinstance(w.data[0,0], np.float64 if precision == "float64" else np.float32)


