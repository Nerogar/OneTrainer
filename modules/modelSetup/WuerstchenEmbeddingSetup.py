from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.modelSetup.mixin.ModelSetupClipEmbeddingMixin import ModelSetupClipEmbeddingMixin
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenEmbeddingSetup(
    BaseWuerstchenSetup,
    ModelSetupClipEmbeddingMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(WuerstchenEmbeddingSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        params += list(model.prior_text_encoder.get_input_embeddings().parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            self.create_param_groups(
                args,
                model.prior_text_encoder.get_input_embeddings().parameters(),
                args.learning_rate,
            ),
        ]

    def setup_model(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        model.prior_text_encoder.requires_grad_(False)
        model.prior_text_encoder.get_input_embeddings().requires_grad_(True)
        model.prior_prior.requires_grad_(False)
        model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

        model.prior_text_encoder.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())

        if len(model.embeddings) == 0:
            vector = self._create_new_embedding(
                model.prior_tokenizer,
                model.prior_text_encoder,
                args.initial_embedding_text,
                args.token_count,
            )

            model.embeddings = [WuerstchenModelEmbedding(vector, 'embedding')]

        original_token_embeds, untrainable_token_ids = self._add_embeddings_to_clip(
            model.prior_tokenizer,
            model.prior_text_encoder,
            [(model.embeddings[0].prior_text_encoder_vector, model.embeddings[0].text_tokens)],
        )
        model.all_prior_text_encoder_original_token_embeds = original_token_embeds
        model.prior_text_encoder_untrainable_token_embeds_mask = untrainable_token_ids

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
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        model.decoder_text_encoder_to(self.temp_device)
        model.decoder_decoder_to(self.temp_device)
        model.decoder_vqgan_to(self.temp_device)
        model.effnet_encoder_to(self.temp_device)

        model.prior_text_encoder_to(self.train_device)
        model.prior_prior_to(self.train_device)

        model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        model.prior_text_encoder.eval()
        model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        self._embeddigns_after_optimizer_step(
            model.prior_text_encoder.get_input_embeddings(),
            model.all_prior_text_encoder_original_token_embeds,
            model.prior_text_encoder_untrainable_token_embeds_mask,
        )
