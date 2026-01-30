from modules.model.QwenModel import QwenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseQwenSetup import BaseQwenSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class QwenFineTuneSetup(
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

        self._create_model_part_parameters(parameter_group_collection, "text_encoder", model.text_encoder, config.text_encoder)
        self._create_model_part_parameters(parameter_group_collection,  "transformer", model.transformer,  config.transformer, freeze=ModuleFilter.create(config), debug=config.debug_mode)

        if config.train_any_embedding() or config.train_any_output_embedding():
            raise NotImplementedError("Embeddings not implemented for Qwen")

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
        self._setup_model_part_requires_grad("text_encoder", model.text_encoder, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)

        model.vae.requires_grad_(False)


    def setup_model(
            self,
            model: QwenModel,
            config: TrainConfig,
    ):
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

factory.register(BaseModelSetup, QwenFineTuneSetup, ModelType.QWEN, TrainingMethod.FINE_TUNE)
