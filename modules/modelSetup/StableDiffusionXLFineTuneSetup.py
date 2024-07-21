import torch

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class StableDiffusionXLFineTuneSetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_1",
                display_name="text_encoder_1",
                parameters=model.text_encoder_1.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.text_encoder_2.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_2",
                display_name="text_encoder_2",
                parameters=model.text_encoder_2.parameters(),
                learning_rate=config.text_encoder_2.learning_rate,
            ))

        if config.train_any_embedding():
            if config.text_encoder.train_embedding:
                for parameter, placeholder, name in zip(model.embedding_wrapper_1.additional_embeddings,
                                                        model.embedding_wrapper_1.additional_embedding_placeholders,
                                                        model.embedding_wrapper_1.additional_embedding_names):
                    parameter_group_collection.add_group(NamedParameterGroup(
                        unique_name=f"embeddings_1/{name}",
                        display_name=f"embeddings_1/{placeholder}",
                        parameters=[parameter],
                        learning_rate=config.embedding_learning_rate,
                    ))

            if config.text_encoder_2.train_embedding:
                for parameter, placeholder, name in zip(model.embedding_wrapper_2.additional_embeddings,
                                                        model.embedding_wrapper_2.additional_embedding_placeholders,
                                                        model.embedding_wrapper_2.additional_embedding_names):
                    parameter_group_collection.add_group(NamedParameterGroup(
                        unique_name=f"embeddings_2/{name}",
                        display_name=f"embeddings_2/{placeholder}",
                        parameters=[parameter],
                        learning_rate=config.embedding_learning_rate,
                    ))

        if config.unet.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="unet",
                display_name="unet",
                parameters=model.unet.parameters(),
                learning_rate=config.unet.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        train_text_encoder_1 = config.text_encoder.train and \
                               not self.stop_text_encoder_training_elapsed(config, model.train_progress)
        model.text_encoder_1.requires_grad_(train_text_encoder_1)

        train_text_encoder_2 = config.text_encoder_2.train and \
                               not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
        model.text_encoder_2.requires_grad_(train_text_encoder_2)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]

            train_embedding = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_1_vector.requires_grad_(train_embedding)

            train_embedding = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)

            embedding.text_encoder_2_vector.requires_grad_(train_embedding)

        train_unet = config.unet.train and \
                     not self.stop_unet_training_elapsed(config, model.train_progress)
        model.unet.requires_grad_(train_unet)

        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop or not config.latent_caching
        text_encoder_1_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.text_encoder_2.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.train()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper_1.normalize_embeddings()
            model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
