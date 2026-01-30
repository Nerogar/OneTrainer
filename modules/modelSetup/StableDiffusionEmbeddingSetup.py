from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class StableDiffusionEmbeddingSetup(
    BaseStableDiffusionSetup,
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
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._add_embedding_param_groups(
            model.all_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
            "embeddings"
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or not config.latent_caching

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
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)

factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_15, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_15_INPAINTING, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_20, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_20_BASE, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_20_INPAINTING, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_20_DEPTH, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_21, TrainingMethod.EMBEDDING)
factory.register(BaseModelSetup, StableDiffusionEmbeddingSetup, ModelType.STABLE_DIFFUSION_21_BASE, TrainingMethod.EMBEDDING)
