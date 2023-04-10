from abc import ABCMeta, abstractmethod


class BaseModelSampler(metaclass=ABCMeta):

    @abstractmethod
    def sample(self, prompt: str, seed: int, destination: str):
        pass
