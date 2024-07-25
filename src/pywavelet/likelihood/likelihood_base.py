from abc import ABC, abstractmethod


class LikelihoodBase(ABC):
    def __init__(self):
        pass

    def log_likelihood(self, data, model):
        pass
