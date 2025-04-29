import copy

from modules.model.HiDreamModel import HiDreamModel
from modules.modelSetup.BaseHiDreamSetup import BaseHiDreamSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class HiDreamFineTuneSetup(
    BaseHiDreamSetup,
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
            model: HiDreamModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_1",
                parameters=model.text_encoder_1.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.text_encoder_2.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_2",
                parameters=model.text_encoder_2.parameters(),
                learning_rate=config.text_encoder_2.learning_rate,
            ))

        if config.text_encoder_3.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_3",
                parameters=model.text_encoder_3.parameters(),
                learning_rate=config.text_encoder_3.learning_rate,
            ))

        if config.text_encoder_4.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_4",
                parameters=model.text_encoder_4.parameters(),
                learning_rate=config.text_encoder_4.learning_rate,
            ))

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder_1 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_1_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding and model.text_encoder_2 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_2_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

            if config.text_encoder_3.train_embedding and model.text_encoder_3 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_3_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_3"
                )

            if config.text_encoder_4.train_embedding and model.text_encoder_4 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_4_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_4"
                )

        if config.prior.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer",
                parameters=model.transformer.parameters(),
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: HiDreamModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)

        if model.text_encoder_1 is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1.requires_grad_(train_text_encoder_1)

        if model.text_encoder_2 is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2.requires_grad_(train_text_encoder_2)

        if model.text_encoder_3 is not None:
            train_text_encoder_3 = config.text_encoder_3.train and \
                                   not self.stop_text_encoder_3_training_elapsed(config, model.train_progress)
            model.text_encoder_3.requires_grad_(train_text_encoder_3)

        if model.text_encoder_4 is not None:
            train_text_encoder_4 = config.text_encoder_4.train and \
                                   not self.stop_text_encoder_4_training_elapsed(config, model.train_progress)
            model.text_encoder_4.requires_grad_(train_text_encoder_4)

        train_transformer = config.prior.train and \
                     not self.stop_prior_training_elapsed(config, model.train_progress)
        model.transformer.requires_grad_(train_transformer)

        model.vae.requires_grad_(False)


    def setup_model(
            self,
            model: HiDreamModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            if model.text_encoder_1 is not None:
                model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_2 is not None:
                model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_3 is not None:
                model.text_encoder_3.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_4 is not None:
                model.text_encoder_4.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        model.tokenizer_1 = copy.deepcopy(model.orig_tokenizer_1)
        model.tokenizer_2 = copy.deepcopy(model.orig_tokenizer_2)
        model.tokenizer_3 = copy.deepcopy(model.orig_tokenizer_3)
        model.tokenizer_4 = copy.deepcopy(model.orig_tokenizer_4)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: HiDreamModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_1_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.train_text_encoder_2_or_embedding() \
            or not config.latent_caching

        text_encoder_3_on_train_device = \
            config.train_text_encoder_3_or_embedding() \
            or not config.latent_caching

        text_encoder_4_on_train_device = \
            config.train_text_encoder_4_or_embedding() \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.text_encoder_3_to(self.train_device if text_encoder_3_on_train_device else self.temp_device)
        model.text_encoder_4_to(self.train_device if text_encoder_4_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder_1:
            if config.text_encoder.train:
                model.text_encoder_1.train()
            else:
                model.text_encoder_1.eval()

        if model.text_encoder_2:
            if config.text_encoder_2.train:
                model.text_encoder_2.train()
            else:
                model.text_encoder_2.eval()

        if model.text_encoder_3:
            if config.text_encoder_3.train:
                model.text_encoder_3.train()
            else:
                model.text_encoder_3.eval()

        if model.text_encoder_4:
            if config.text_encoder_4.train:
                model.text_encoder_4.train()
            else:
                model.text_encoder_4.eval()

        model.vae.eval()

        if config.prior.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: HiDreamModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_3_embeddings())
            self._normalize_output_embeddings(model.all_text_encoder_4_embeddings())
            if model.embedding_wrapper_1 is not None:
                model.embedding_wrapper_1.normalize_embeddings()
            if model.embedding_wrapper_2 is not None:
                model.embedding_wrapper_2.normalize_embeddings()
            if model.embedding_wrapper_3 is not None:
                model.embedding_wrapper_3.normalize_embeddings()
            if model.embedding_wrapper_4 is not None:
                model.embedding_wrapper_4.normalize_embeddings()
        self.__setup_requires_grad(model, config)
