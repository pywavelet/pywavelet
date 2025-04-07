import logging
import sys
import warnings

from rich.logging import RichHandler

FORMAT = "%(message)s"

logger = logging.getLogger("pywavelet")
if not logger.hasHandlers():
    handler = RichHandler(rich_tracebacks=True)
    formatter = logging.Formatter(fmt=FORMAT, datefmt="[%X]")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
