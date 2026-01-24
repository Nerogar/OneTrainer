from modules.model.HiDreamModel import HiDreamModel
from modules.modelSetup.BaseHiDreamSetup import BaseHiDreamSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class HiDreamEmbeddingSetup(
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

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: HiDreamModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        if model.text_encoder_1 is not None:
            model.text_encoder_1.requires_grad_(False)
        if model.text_encoder_2 is not None:
            model.text_encoder_2.requires_grad_(False)
        if model.text_encoder_3 is not None:
            model.text_encoder_3.requires_grad_(False)
        if model.text_encoder_4 is not None:
            model.text_encoder_4.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: HiDreamModel,
            config: TrainConfig,
    ):
        if model.text_encoder_1 is not None:
            model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        if model.text_encoder_2 is not None:
            model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        if model.text_encoder_3 is not None:
            model.text_encoder_3.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
        if model.text_encoder_4 is not None:
            model.text_encoder_4.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

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

        model.text_encoder_1_to(self.train_device if config.text_encoder.train_embedding else self.temp_device)
        model.text_encoder_2_to(self.train_device if config.text_encoder_2.train_embedding else self.temp_device)
        model.text_encoder_3_to(self.train_device if config.text_encoder_3.train_embedding else self.temp_device)
        model.text_encoder_4_to(self.train_device if config.text_encoder_4.train_embedding else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder_1 is not None:
            model.text_encoder_1.eval()
        if model.text_encoder_2 is not None:
            model.text_encoder_2.eval()
        if model.text_encoder_3 is not None:
            model.text_encoder_3.eval()
        if model.text_encoder_4 is not None:
            model.text_encoder_4.eval()
        model.vae.eval()
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

factory.register(BaseModelSetup, HiDreamEmbeddingSetup, ModelType.HI_DREAM_FULL, TrainingMethod.EMBEDDING)
