import torch

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class WuerstchenEmbeddingSetup(
    BaseWuerstchenSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(WuerstchenEmbeddingSetup, self).__init__(
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

        for parameter, placeholder, name in zip(model.prior_embedding_wrapper.additional_embeddings,
                                                model.prior_embedding_wrapper.additional_embedding_placeholders,
                                                model.prior_embedding_wrapper.additional_embedding_names):
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"prior_embeddings/{name}",
                display_name=f"prior_embeddings/{placeholder}",
                parameters=[parameter],
                learning_rate=config.embedding_learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        model.prior_text_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)
        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

        model.embedding.prior_text_encoder_vector.requires_grad_(True)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = \
                embedding_config.train \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.prior_text_encoder_vector.requires_grad_(train_embedding)

    def setup_model(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        model.prior_text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.prior_tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self.setup_optimizations(model, config)

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

        model.prior_text_encoder_to(self.train_device)
        model.prior_prior_to(self.train_device)

        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        model.prior_text_encoder.eval()
        model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.prior_embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
