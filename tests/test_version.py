import pywavelet


def test_version():
    assert hasattr(pywavelet, "__version__")
    assert isinstance(pywavelet.__version__, str)
