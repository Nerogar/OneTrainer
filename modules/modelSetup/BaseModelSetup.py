from abc import ABCMeta, abstractmethod
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from modules.model.BaseModel import BaseModel
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.enum.LearningRateScaler import LearningRateScaler


class BaseModelSetup(
    TimedActionMixin,
    metaclass=ABCMeta,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(BaseModelSetup, self).__init__()

        self.train_device = train_device
        self.temp_device = temp_device
        self.debug_mode = debug_mode

    @abstractmethod
    def create_parameters(
            self,
            model: BaseModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        pass

    @abstractmethod
    def setup_model(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def setup_train_device(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
        pass

    @abstractmethod
    def predict(
            self,
            model: BaseModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        pass

    @abstractmethod
    def calculate_loss(
            self,
            model: BaseModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        pass

    @abstractmethod
    def after_optimizer_step(
            self,
            model: BaseModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        pass

    @abstractmethod
    def report_learning_rates(
            self,
            model: BaseModel,
            config: TrainConfig,
            scheduler: LRScheduler,
            tensorboard: SummaryWriter
    ):
        """Reports current learning rates per the scheduler to tensorboard."""
        pass

    def stop_unet_training_elapsed(
            self,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_unet_training",
            config.unet.stop_training_after,
            config.unet.stop_training_after_unit,
            train_progress,
        )

    def stop_prior_training_elapsed(
            self,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_prior_training",
            config.prior.stop_training_after,
            config.prior.stop_training_after_unit,
            train_progress,
        )

    def stop_text_encoder_training_elapsed(
            self,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_text_encoder_training",
            config.text_encoder.stop_training_after,
            config.text_encoder.stop_training_after_unit,
            train_progress,
        )

    def stop_text_encoder_2_training_elapsed(
            self,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_text_encoder_2_training",
            config.text_encoder_2.stop_training_after,
            config.text_encoder_2.stop_training_after_unit,
            train_progress,
        )

    def stop_additional_embedding_training_elapsed(
            self,
            config: TrainEmbeddingConfig,
            train_progress: TrainProgress,
            embedding_index: int,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training_" + str(embedding_index),
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    def stop_embedding_training_elapsed(
            self,
            config: TrainEmbeddingConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training",
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )
