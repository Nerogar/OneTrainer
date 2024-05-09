import torch

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class StableDiffusionXLEmbeddingSetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLEmbeddingSetup, self).__init__(
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

        for parameter, placeholder, name in zip(model.embedding_wrapper_1.additional_embeddings,
                                                model.embedding_wrapper_1.additional_embedding_placeholders,
                                                model.embedding_wrapper_1.additional_embedding_names):
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"embeddings_1/{name}",
                display_name=f"embeddings_1/{placeholder}",
                parameters=[parameter],
                learning_rate=config.embedding_learning_rate,
            ))

        for parameter, placeholder, name in zip(model.embedding_wrapper_2.additional_embeddings,
                                                model.embedding_wrapper_2.additional_embedding_placeholders,
                                                model.embedding_wrapper_2.additional_embedding_names):
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"embeddings_2/{name}",
                display_name=f"embeddings_2/{placeholder}",
                parameters=[parameter],
                learning_rate=config.embedding_learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.embedding.text_encoder_1_vector.requires_grad_(True)
        model.embedding.text_encoder_2_vector.requires_grad_(True)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = \
                embedding_config.train \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_1_vector.requires_grad_(train_embedding)
            embedding.text_encoder_2_vector.requires_grad_(train_embedding)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding(model, config)
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

        model.text_encoder_to(self.train_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        model.text_encoder_1.eval()
        model.text_encoder_2.eval()
        model.vae.eval()
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
