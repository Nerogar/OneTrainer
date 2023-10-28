from abc import ABCMeta, abstractmethod

import torch
from mgds.MGDS import MGDS, TrainDataLoader

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.model.BaseModel import BaseModel


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super(BaseDataLoader, self).__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def get_data_set(self) -> MGDS:
        pass

    @abstractmethod
    def get_data_loader(self) -> TrainDataLoader:
        pass

    @abstractmethod
    def setup_cache_device(
            self,
            model: BaseModel,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        pass
