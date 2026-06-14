from modules.model.IdeogramModel import IdeogramModel
from modules.modelSetup.BaseIdeogramSetup import BaseIdeogramSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class IdeogramFineTuneSetup(
    BaseIdeogramSetup,
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
            model: IdeogramModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer, config.transformer,
            freeze=ModuleFilter.create(config), debug=config.debug_mode,
        )
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        # the unconditional transformer is never trained (only the negative branch of the dual-network CFG at
        # sampling), and may not be loaded at all
        if model.unconditional_transformer is not None:
            model.unconditional_transformer.requires_grad_(False)

    def setup_model(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.image_caching
        text_encoder_on_train_device = not config.text_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)
        # the unconditional transformer is only needed for sampling; keep it off the train device during training
        model.unconditional_transformer_to(self.temp_device)

        model.text_encoder.eval()
        model.vae.eval()
        if model.unconditional_transformer is not None:
            model.unconditional_transformer.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: IdeogramModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, IdeogramFineTuneSetup, ModelType.IDEOGRAM_4, TrainingMethod.FINE_TUNE)
