from ..logger import logger
from .coupled_data import CoupledData


class Data(CoupledData):
    logger.warning(
        "The Data class is deprecated and will be removed in a future release. "
        "Please use CoupledData instead."
    )
