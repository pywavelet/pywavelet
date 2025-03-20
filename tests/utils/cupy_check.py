import importlib.util

# Check if CuPy is available and CUDA is accessible
cupy_available = importlib.util.find_spec("cupy") is not None
if cupy_available:
    import cupy

    try:
        cupy.cuda.runtime.getDeviceCount()  # Check if any CUDA device is available
        cuda_available = True
    except cupy.cuda.runtime.CUDARuntimeError:
        cuda_available = False
else:
    cuda_available = False
