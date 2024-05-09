import torch

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class StableDiffusionEmbeddingSetup(
    BaseStableDiffusionSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionEmbeddingSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        for parameter, placeholder, name in zip(model.embedding_wrapper.additional_embeddings,
                                                model.embedding_wrapper.additional_embedding_placeholders,
                                                model.embedding_wrapper.additional_embedding_names):
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"embeddings/{name}",
                display_name=f"embeddings/{placeholder}",
                parameters=[parameter],
                learning_rate=config.embedding_learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.embedding.text_encoder_vector.requires_grad_(True)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = \
                embedding_config.train \
                and not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_vector.requires_grad_(train_embedding)

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop_loss or not config.latent_caching

        model.text_encoder_to(self.train_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        model.text_encoder.eval()
        model.vae.eval()
        model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
