from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
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
        self.frozen_parameters = {}

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

    def __stop_model_part_training_elapsed(
            self,
            unique_name: str,
            config: TrainModelPartConfig,
            train_progress: TrainProgress,
    ):
        return self.single_action_elapsed(
            "stop_" + unique_name + "_training",
            config.stop_training_after,
            config.stop_training_after_unit,
            train_progress,
        )

    @contextmanager
    def prior_model(self, model: BaseModel, config: TrainConfig):
        if config.training_method is not TrainingMethod.LORA:
            raise NotImplementedError("Prior model is only available with LoRA training")

        for adapter in model.adapters():
            adapter.remove_hook_from_module()
        try:
            yield
        finally:
            for adapter in model.adapters():
                adapter.hook_to_module()

    def _create_layer_key_fn(self, model: BaseModel) -> callable:
        from modules.util.layer_identifier import build_layer_identifier_fn
        return build_layer_identifier_fn(model, self.debug_mode)

    def _create_model_part_parameters(
        self,
        parameter_group_collection: NamedParameterGroupCollection,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        freeze: list[ModuleFilter] | None = None,
        debug: bool = False,
    ):
        if not config.train:
            return

        if freeze is not None and len(freeze) > 0:
            selected = []
            deselected = []
            parameters = []
            self.frozen_parameters[unique_name] = []
            for name, param in model.named_parameters():
                if any(f.matches(name) for f in freeze):
                    parameters.append(param)
                    selected.append(name)
                else:
                    self.frozen_parameters[unique_name].append(param)
                    deselected.append(name)

            if debug:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")
        else:
            parameters = model.parameters()

        parameter_group_collection.add_group(NamedParameterGroup(
            unique_name=unique_name,
            parameters=parameters,
            learning_rate=config.learning_rate,
        ))

    def _setup_model_part_requires_grad(
        self,
        unique_name: str,
        model: torch.nn.Module,
        config: TrainModelPartConfig,
        train_progress: TrainProgress,
    ):
        if model is not None:
            train_model_part = config.train and \
                               not self.__stop_model_part_training_elapsed(unique_name, config, train_progress)
            model.requires_grad_(train_model_part)

            #even if frozen parameters are not passed to the optimizer, required_grad has to be False.
            #otherwise, gradients accumulate in param.grad and waste vram
            if unique_name in self.frozen_parameters:
                for param in self.frozen_parameters[unique_name]:
                    param.requires_grad_(False)
