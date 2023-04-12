from abc import ABCMeta, abstractmethod


class BaseModelSampler(metaclass=ABCMeta):

    @abstractmethod
    def sample(self, prompt: str, resolution: tuple[int, int], seed: int, destination: str):
        pass
