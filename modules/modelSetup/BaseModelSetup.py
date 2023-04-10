from abc import ABCMeta, abstractmethod

import torch

from modules.model.BaseModel import BaseModel


class BaseModelSetup(metaclass=ABCMeta):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def start_data_loader(self, model: BaseModel):
        pass

    @abstractmethod
    def start_train(self, model: BaseModel, train_text_encoder: bool):
        pass

    @abstractmethod
    def start_eval(self, model: BaseModel):
        pass
