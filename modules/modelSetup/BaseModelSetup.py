from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


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
        super().__init__()

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
    def setup_optimizations(
            self,
            model: BaseModel,
            config: TrainConfig,
    ):
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

    def report_to_tensorboard(
            self,
            model: BaseModel,
            config: TrainConfig,
            scheduler: LRScheduler,
            tensorboard: SummaryWriter,
    ):
        lrs = scheduler.get_last_lr()
        parameters = model.parameters.display_name_mapping

        reported_learning_rates = {}
        for lr, parameter in zip(lrs, parameters, strict=True):
            # only use the prefix. this prevents multiple embedding reports. TODO: find a better solution
            name = parameter.split('/')[0]

            if name not in reported_learning_rates:
                reported_learning_rates[name] = lr

        reported_learning_rates = config.optimizer.optimizer.maybe_adjust_lrs(reported_learning_rates, model.optimizer)

        for name, lr in reported_learning_rates.items():
            tensorboard.add_scalar(
                f"lr/{name}", lr, model.train_progress.global_step
            )

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

    def stop_text_encoder_3_training_elapsed(
            self,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_text_encoder_3_training",
            config.text_encoder_3.stop_training_after,
            config.text_encoder_3.stop_training_after_unit,
            train_progress,
        )

    def stop_embedding_training_elapsed(
            self,
            config: TrainEmbeddingConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_embedding_training_" + str(config.uuid),
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    @contextmanager
    def prior_model(self, model: BaseModel, config: TrainConfig):
        if (config.training_method is not TrainingMethod.LORA
            or config.train_any_embedding()
            or config.train_any_output_embedding()):
            raise NotImplementedError("Prior model is only available with LoRA training and if no embeddings are trained")

        for adapter in model.adapters():
            adapter.remove_hook_from_module()
        try:
            yield
        finally:
            for adapter in model.adapters():
                adapter.hook_to_module()
