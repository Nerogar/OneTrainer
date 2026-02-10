from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig


class ModelSetupText2ImageMixin(metaclass=ABCMeta):
    @abstractmethod
    def prepare_text_caching(self, model: BaseModel, config: TrainConfig):
        pass

    #for future use in samplers etc.
    '''@abstractmethod
    def prepare_training(self, model: BaseModel):
        pass

    @abstractmethod
    def prepare_text_inference(self, model: BaseModel):
        pass

    @abstractmethod
    def prepare_image_inference(self, model: BaseModel):
        pass'''
