from modules.model.LensModel import LensModel
from modules.modelSetup.BaseLensSetup import BaseLensSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class LensLoRASetup(
    BaseLensSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: LensModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "transformer", model.transformer_lora, config.transformer)
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: LensModel,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: LensModel,
            config: TrainConfig,
    ):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: LensModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.image_caching
        text_encoder_on_train_device = not config.text_caching

        # the encoder is always on-demand for Lens, so materialize it when training needs it and
        # discard it (freeing VRAM) otherwise -- text_encoder_to would be a no-op on the proxy.
        if text_encoder_on_train_device:
            model.materialize_text_encoder(self.train_device)
        else:
            model.release_text_encoder()
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder is not None:
            model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: LensModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, LensLoRASetup, ModelType.LENS, TrainingMethod.LORA)
