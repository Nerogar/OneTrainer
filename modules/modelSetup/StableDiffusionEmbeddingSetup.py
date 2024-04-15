from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.modelSetup.mixin.ModelSetupClipEmbeddingMixin import ModelSetupClipEmbeddingMixin
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionEmbeddingSetup(
    BaseStableDiffusionSetup,
    ModelSetupClipEmbeddingMixin,
):
    all_original_token_embeds: Tensor
    trainable_token_embeds_mask: list[bool]
    untrainable_token_embeds_mask: list[bool]

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
    ) -> Iterable[Parameter]:
        return model.text_encoder.get_input_embeddings().parameters()

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            self.create_param_groups(
                config,
                model.text_encoder.get_input_embeddings().parameters(),
                config.learning_rate,
            )
        ]

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.text_encoder.get_input_embeddings().requires_grad_(True)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if len(model.embeddings) == 0:
            vector = self._create_new_embedding(
                model.tokenizer,
                model.text_encoder,
                config.embeddings[0].initial_embedding_text,
                config.embeddings[0].token_count,
            )

            model.embeddings = [StableDiffusionModelEmbedding(vector, 'embedding')]

        original_token_embeds, untrainable_token_ids = self._add_embeddings_to_clip(
            model.tokenizer,
            model.text_encoder,
            [(model.embeddings[0].text_encoder_vector, model.embeddings[0].text_tokens)],
        )
        model.all_text_encoder_original_token_embeds = original_token_embeds
        model.text_encoder_untrainable_token_embeds_mask = untrainable_token_ids

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, config), model.optimizer_state_dict, config
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, config), model.ema_state_dict, config
        )
        del model.ema_state_dict

        self.setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop_loss or not config.latent_caching

        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet.to(self.train_device)
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
        self._embeddigns_after_optimizer_step(
            model.text_encoder.get_input_embeddings(),
            model.all_text_encoder_original_token_embeds,
            model.text_encoder_untrainable_token_embeds_mask,
        )

    def report_learning_rates(
            self,
            model,
            config,
            scheduler,
            tensorboard
    ):
        lr = scheduler.get_last_lr()[0]
        lr = config.optimizer.optimizer.maybe_adjust_lrs([lr], model.optimizer)[0]
        tensorboard.add_scalar("lr/embedding", lr, model.train_progress.global_step)
