from modules.model.MossModel import MossModel
from modules.modelSetup.BaseMossSetup import BaseMossSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress
import torch


@factory.register(BaseModelSetup, ModelType.MOSS, TrainingMethod.EMBEDDING)
class MossEmbeddingSetup(
    BaseMossSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(train_device, temp_device, debug_mode)
        self._conditioner_output_std: float | None = None

    def _setup_embeddings_requires_grad(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            model.append_tokens[embedding.uuid].requires_grad_(train_embedding)

    def create_parameters(
            self,
            model: MossModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train_embedding:
            for embedding, embedding_config in zip(model.all_embeddings(),
                                                    config.all_embedding_configs(), strict=True):
                if not embedding_config.train:
                    continue
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings/{embedding.uuid}",
                    display_name=f"embeddings/{embedding.placeholder}",
                    parameters=[model.append_tokens[embedding.uuid]],
                    learning_rate=config.embedding_learning_rate,
                ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        self._setup_embeddings(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        parts = ["transformer"]
        if text_encoder_on_train_device:
            parts.append("text_encoder")
        if vae_on_train_device:
            parts.append("vae")
        model.materialize_only(*parts)

        model.text_encoder.eval()
        model.vae.eval()
        model.transformer.eval()

    def prepare_text_caching(self, model: MossModel, config: TrainConfig):
        super().prepare_text_caching(model, config)
        if config.preserve_embedding_norm and self._conditioner_output_std is None:
            self._measure_conditioner_output_std(model)

    def after_optimizer_step(
            self,
            model: MossModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_append_tokens(model)
        self.__setup_requires_grad(model, config)

    def _normalize_append_tokens(self, model: MossModel):
        with torch.no_grad():
            target_std = self._get_conditioner_output_std(model)
            if target_std <= 0:
                return
            for embedding in model.all_embeddings():
                token_tensor = model.append_tokens.get(embedding.uuid)
                if token_tensor is not None and token_tensor.requires_grad:
                    std = token_tensor.std(dim=1).mean()
                    if std > 0:
                        token_tensor.mul_(target_std / std)

    def _measure_conditioner_output_std(self, model: MossModel) -> float:
        if self._conditioner_output_std is None:
            try:
                # Text encoder can only run forward if it is on CUDA/train_device
                if next(model.text_encoder.parameters()).device == self.train_device:
                    with torch.no_grad(), model.autocast_context:
                        # MOSS text output is 3D (B, T, H)
                        dummy_output, _ = model.encode_text(
                            text="*",
                            train_device=self.train_device,
                            apply_append_tokens=False,
                        )
                        self._conditioner_output_std = dummy_output.std(dim=(1, 2)).mean().item()
                else:
                    self._conditioner_output_std = 1.0
            except Exception:
                self._conditioner_output_std = 1.0
        return self._conditioner_output_std

    def _get_conditioner_output_std(self, model: MossModel) -> float:
        if self._conditioner_output_std is None:
            return self._measure_conditioner_output_std(model)
        return self._conditioner_output_std
