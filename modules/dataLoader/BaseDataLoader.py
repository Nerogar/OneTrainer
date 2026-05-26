import copy
from abc import ABCMeta, abstractmethod

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader

import torch


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        self.__ds = self._create_dataset(
            config=config,
            model=model,
            model_setup=model_setup,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    @abstractmethod
    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation,
    ):
        pass
