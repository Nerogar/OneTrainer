from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class StableDiffusionXLEmbeddingSetup(
    BaseStableDiffusionXLSetup,
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
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train_embedding:
            self._add_embedding_param_groups(
                model.all_text_encoder_1_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                "embeddings_1"
            )

        if config.text_encoder_2.train_embedding:
            self._add_embedding_param_groups(
                model.all_text_encoder_2_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                "embeddings_2"
            )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching

        model.text_encoder_1_to(self.train_device if config.text_encoder.train_embedding else self.temp_device)
        model.text_encoder_2_to(self.train_device if config.text_encoder_2.train_embedding else self.temp_device)
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
            self._normalize_output_embeddings(model.all_text_encoder_1_embeddings())
            self._normalize_output_embeddings(model.all_text_encoder_2_embeddings())
            model.embedding_wrapper_1.normalize_embeddings()
            model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, StableDiffusionXLEmbeddingSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionXLEmbeddingSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING, TrainingMethod.EMBEDDING)
