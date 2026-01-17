from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig


class ModelSetupText2ImageMixin(metaclass=ABCMeta):
    @abstractmethod
    def prepare_text_caching(model: BaseModel, config: TrainConfig):
        pass

    #for future use in samplers etc.
    '''@abstractmethod
    def prepare_training(model: BaseModel):
        pass

    @abstractmethod
    def prepare_text_inference(model: BaseModel):
        pass

    @abstractmethod
    def prepare_image_inference(model: BaseModel):
        pass'''
