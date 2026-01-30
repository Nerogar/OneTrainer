from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class WuerstchenFineTuneSetup(
    BaseWuerstchenSetup,
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
            model: WuerstchenModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "prior_text_encoder", model.prior_text_encoder, config.text_encoder)

        if config.train_any_embedding() or config.train_any_output_embedding():
            self._add_embedding_param_groups(
                model.all_prior_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                "prior_embeddings"
            )

        self._create_model_part_parameters(parameter_group_collection, "prior_prior", model.prior_prior, config.prior,
                                           freeze=ModuleFilter.create(config), debug=config.debug_mode)

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder", model.text_encoder, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("prior_prior", model.prior_prior, config.prior, model.train_progress)

    def setup_model(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.prior_text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.prior_tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        effnet_on_train_device = not config.latent_caching

        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder_to(self.temp_device)
        model.decoder_decoder_to(self.temp_device)
        model.decoder_vqgan_to(self.temp_device)
        model.effnet_encoder_to(self.train_device if effnet_on_train_device else self.temp_device)

        text_encoder_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or not config.latent_caching

        model.prior_text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.prior_prior_to(self.train_device)

        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        if config.text_encoder.train:
            model.prior_text_encoder.train()
        else:
            model.prior_text_encoder.eval()

        if config.prior.train:
            model.prior_prior.train()
        else:
            model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_prior_text_encoder_embeddings())
            model.prior_embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, WuerstchenFineTuneSetup, ModelType.WUERSTCHEN_2, TrainingMethod.FINE_TUNE)
factory.register(BaseModelSetup, WuerstchenFineTuneSetup, ModelType.STABLE_CASCADE_1, TrainingMethod.FINE_TUNE)
