from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.modelSetup.mixin.ModelSetupClipEmbeddingMixin import ModelSetupClipEmbeddingMixin
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLEmbeddingSetup(
    BaseStableDiffusionXLSetup,
    ModelSetupClipEmbeddingMixin,
):
    all_text_encoder_1_original_token_embeds: Tensor
    text_encoder_1_trainable_token_embeds_mask: list[bool]
    text_encoder_1_untrainable_token_embeds_mask: list[bool]

    all_text_encoder_2_original_token_embeds: Tensor
    text_encoder_2_trainable_token_embeds_mask: list[bool]
    text_encoder_2_untrainable_token_embeds_mask: list[bool]

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
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        params += list(model.text_encoder_1.get_input_embeddings().parameters())
        params += list(model.text_encoder_2.get_input_embeddings().parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            self.create_param_groups(
                args,
                model.text_encoder_1.get_input_embeddings().parameters(),
                args.learning_rate,
            ),
            self.create_param_groups(
                args,
                model.text_encoder_2.get_input_embeddings().parameters(),
                args.learning_rate,
            )
        ]

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.text_encoder_1.get_input_embeddings().requires_grad_(True)
        model.text_encoder_2.get_input_embeddings().requires_grad_(True)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.text_encoder_1.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())
        model.text_encoder_2.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())

        if len(model.embeddings) == 0:
            vector_1 = self._create_new_embedding(
                model.tokenizer_1,
                model.text_encoder_1,
                args.initial_embedding_text,
                args.token_count,
            )

            vector_2 = self._create_new_embedding(
                model.tokenizer_2,
                model.text_encoder_2,
                args.initial_embedding_text,
                args.token_count,
            )

            model.embeddings = [StableDiffusionXLModelEmbedding(vector_1, vector_2, 'embedding')]

        original_token_embeds_1, untrainable_token_ids_1 = self._add_embeddings_to_clip(
            model.tokenizer_1,
            model.text_encoder_1,
            [(model.embeddings[0].text_encoder_1_vector, model.embeddings[0].text_tokens, True)],
        )
        model.all_text_encoder_1_original_token_embeds = original_token_embeds_1
        model.text_encoder_1_untrainable_token_embeds_mask = untrainable_token_ids_1

        original_token_embeds_2, untrainable_token_ids_2 = self._add_embeddings_to_clip(
            model.tokenizer_2,
            model.text_encoder_2,
            [(model.embeddings[0].text_encoder_2_vector, model.embeddings[0].text_tokens, True)],
        )
        model.all_text_encoder_2_original_token_embeds = original_token_embeds_2
        model.text_encoder_2_untrainable_token_embeds_mask = untrainable_token_ids_2

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, args), model.optimizer_state_dict, args
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, args), model.ema_state_dict, args
        )
        del model.ema_state_dict

        self.setup_optimizations(model, args)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        vae_on_train_device = args.align_prop

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
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        self._embeddigns_after_optimizer_step(
            model.text_encoder_1.get_input_embeddings(),
            model.all_text_encoder_1_original_token_embeds,
            model.text_encoder_1_untrainable_token_embeds_mask,
        )

        self._embeddigns_after_optimizer_step(
            model.text_encoder_2.get_input_embeddings(),
            model.all_text_encoder_2_original_token_embeds,
            model.text_encoder_2_untrainable_token_embeds_mask,
        )
