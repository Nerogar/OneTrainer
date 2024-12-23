import copy

from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSetup.BaseHunyuanVideoSetup import BaseHunyuanVideoSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class HunyuanVideoEmbeddingSetup(
    BaseHunyuanVideoSetup,
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
            model: HunyuanVideoModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train_embedding and model.text_encoder_1 is not None:
            self._add_embedding_param_groups(
                model.embedding_wrapper_1, parameter_group_collection, config.embedding_learning_rate, "embeddings_1"
            )

        if config.text_encoder_2.train_embedding and model.text_encoder_2 is not None:
            self._add_embedding_param_groups(
                model.embedding_wrapper_2, parameter_group_collection, config.embedding_learning_rate, "embeddings_2"
            )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        if model.text_encoder_1 is not None:
            model.text_encoder_1.requires_grad_(False)
        if model.text_encoder_2 is not None:
            model.text_encoder_2.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.transformer.requires_grad_(False)

        if model.text_encoder_1 is not None:
            model.embedding.text_encoder_1_vector.requires_grad_(config.text_encoder.train_embedding)
        if model.text_encoder_2 is not None:
            model.embedding.text_encoder_2_vector.requires_grad_(config.text_encoder_2.train_embedding)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            if model.text_encoder_1 is not None:
                train_embedding_1 = \
                    embedding_config.train \
                    and config.text_encoder.train_embedding \
                    and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
                embedding.text_encoder_1_vector.requires_grad_(train_embedding_1)
            if model.text_encoder_2 is not None:
                train_embedding_2 = \
                    embedding_config.train \
                    and config.text_encoder.train_embedding \
                    and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
                embedding.text_encoder_2_vector.requires_grad_(train_embedding_2)

    def setup_model(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        if model.text_encoder_1 is not None:
            model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        if model.text_encoder_2 is not None:
            model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        model.tokenizer_1 = copy.deepcopy(model.orig_tokenizer_1)
        model.tokenizer_2 = copy.deepcopy(model.orig_tokenizer_2)
        self._setup_embeddings(model, config)
        self._setup_embedding(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop or not config.latent_caching

        model.text_encoder_1_to(self.train_device if config.text_encoder.train_embedding else self.temp_device)
        model.text_encoder_2_to(self.train_device if config.text_encoder_2.train_embedding else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder_1 is not None:
            model.text_encoder_1.eval()
        if model.text_encoder_2 is not None:
            model.text_encoder_2.eval()
        model.vae.eval()
        model.transformer.eval()

    def after_optimizer_step(
            self,
            model: HunyuanVideoModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            if model.embedding_wrapper_1 is not None:
                model.embedding_wrapper_1.normalize_embeddings()
            if model.embedding_wrapper_2 is not None:
                model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
