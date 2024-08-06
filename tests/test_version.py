import pywavelet


def test_version():
    assert hasattr(
        pywavelet, "__version__"
    ), "pywavelet has no __version__ attribute"
    assert isinstance(
        pywavelet.__version__, str
    ), f"{pywavelet.__version__} is not a string"
    assert pywavelet.__version__ > "0.0.0", f"{pywavelet.__version__} <= 0.0.0"
