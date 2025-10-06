import itertools
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig, TrainModelPartConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.nn import Module
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

    def _create_kourkoutas_layer_key_fn(self, model: BaseModel) -> callable:
        """
        Creates a function that maps a parameter to its layer's string identifier
        based on its full name. Correctly groups parameters.
        """
        print("[Kourkoutas-β Debug] Starting to build layer key function...")

        param_map = {}

        sub_modules_to_check = {
            'text_encoder': getattr(model, 'text_encoder', None),
            'text_encoder_1': getattr(model, 'text_encoder_1', None),
            'text_encoder_2': getattr(model, 'text_encoder_2', None),
            'text_encoder_3': getattr(model, 'text_encoder_3', None),
            'text_encoder_4': getattr(model, 'text_encoder_4', None),
            'unet': getattr(model, 'unet', None),
            'transformer': getattr(model, 'transformer', None),
            'text_encoder_lora': getattr(model, 'text_encoder_lora', None),
            'text_encoder_1_lora': getattr(model, 'text_encoder_1_lora', None),
            'text_encoder_2_lora': getattr(model, 'text_encoder_2', None),
            'text_encoder_3_lora': getattr(model, 'text_encoder_3', None),
            'text_encoder_4_lora': getattr(model, 'text_encoder_4', None),
            'unet_lora': getattr(model, 'unet_lora', None),
            'transformer_lora': getattr(model, 'transformer_lora', None),
        }

        for module_name, module in sub_modules_to_check.items():
            if module is None:
                continue

            count = 0
            if isinstance(module, LoRAModuleWrapper):
                for lora_module in module.lora_modules.values():
                    layer_key = lora_module.prefix.rstrip('.')
                    for _param_name, p in lora_module.named_parameters():
                        if p.requires_grad:
                            param_map[id(p)] = layer_key
                            count += 1
            elif isinstance(module, Module):
                for param_name, p in module.named_parameters():
                    if p.requires_grad:
                        # For standard modules, group by the module path
                        full_name = f"{module_name}.{param_name}"
                        param_map[id(p)] = full_name.rpartition('.')[0]
                        count += 1

            if count > 0:
                print(f"[Kourkoutas-β Debug] Scanned '{module_name}', found {count} trainable parameters.")

        if hasattr(model, 'all_text_encoder_embeddings'):
            count = 0
            for emb_container in model.all_text_encoder_embeddings():
                if isinstance(emb_container, BaseModelEmbedding) and emb_container.vector is not None and emb_container.vector.requires_grad:
                    # Each embedding is its own bucket
                    full_name = f"embedding.{emb_container.placeholder}"
                    param_map[id(emb_container.vector)] = full_name
                    count += 1
                elif isinstance(emb_container, Module):
                     for param_name, p in emb_container.named_parameters():
                        if p.requires_grad:
                            full_name = f"embedding_module.{param_name}"
                            param_map[id(p)] = full_name.rpartition('.')[0]
                            count += 1
            if count > 0:
                print(f"[Kourkoutas-β Debug] Scanned embeddings, found {count} trainable embedding vectors.")

        print(f"[Kourkoutas-β Debug] Total trainable parameters mapped: {len(param_map)}")
        if len(param_map) > 0:
            print("[Kourkoutas-β Debug] Sample of mapped parameter names:")
            for _i, (_p_id, name) in enumerate(itertools.islice(param_map.items(), 5)):
                print(f"  - Mapped to bucket key: '{name}'")
            if len(param_map) > 5:
                print("  - ...")

        def layer_key_fn(p: torch.nn.Parameter) -> str:
            layer_key = param_map.get(id(p))
            if layer_key is None:
                return 'unmapped_param'
            return layer_key

        return layer_key_fn

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
