from modules.model.QwenModel import QwenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseQwenSetup import BaseQwenSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch


class QwenLoRASetup(
    BaseQwenSetup,
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
            model: QwenModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "text_encoder", model.text_encoder_lora, config.text_encoder)
        self._create_model_part_parameters(parameter_group_collection, "transformer",  model.transformer_lora,  config.transformer)

        if config.train_any_embedding() or config.train_any_output_embedding():
            raise NotImplementedError("Embeddings not implemented for Qwen")

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
        if model.text_encoder is not None:
            model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder", model.text_encoder_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "text_encoder")

        if model.text_encoder is not None:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, "text_encoder", config
            ) if create_te else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: QwenModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, QwenLoRASetup, ModelType.QWEN, TrainingMethod.LORA)
