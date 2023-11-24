from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModelEmbedding
from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenEmbeddingSetup(BaseWuerstchenSetup):
    all_prior_text_encoder_original_token_embeds: Tensor
    prior_text_encoder_trainable_token_embeds_mask: list[bool]
    prior_text_encoder_untrainable_token_embeds_mask: list[bool]

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
            {
                'params': model.prior_text_encoder.get_input_embeddings().parameters(),
                'lr': args.learning_rate,
                'initial_lr': args.learning_rate,
            }
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

        token_count = args.token_count if len(model.embeddings) == 0 else model.embeddings[0].token_count

        tokens = [f"<embedding_{i}>" for i in range(token_count)]
        model.prior_tokenizer.add_tokens(tokens)
        model.prior_text_encoder.resize_token_embeddings(len(model.prior_tokenizer))

        model.prior_text_encoder.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())

        with torch.no_grad():
            prior_text_encoder_token_ids = model.prior_tokenizer.encode(
                tokens,
                add_special_tokens=False,
            )

            all_prior_text_encoder_token_embeds = model.prior_text_encoder.get_input_embeddings().weight.data
            self.all_prior_text_encoder_original_token_embeds = all_prior_text_encoder_token_embeds.clone()
            self.prior_text_encoder_trainable_token_embeds_mask = \
                [(i in prior_text_encoder_token_ids) for i in
                 range(len(self.all_prior_text_encoder_original_token_embeds))]
            self.prior_text_encoder_untrainable_token_embeds_mask = [
                (i not in prior_text_encoder_token_ids) for i in
                range(len(self.all_prior_text_encoder_original_token_embeds))
            ]

            if len(model.embeddings) > 0:
                # an embedding was loaded
                for i, token_id in enumerate(prior_text_encoder_token_ids):
                    all_prior_text_encoder_token_embeds[token_id] = model.embeddings[0].prior_text_encoder_vector[i]
            else:
                # create a new embedding
                prior_text_encoder_initial_token_ids = model.prior_tokenizer.encode(
                    args.initial_embedding_text,
                    add_special_tokens=False,
                    max_length=token_count,
                )
                prior_text_encoder_pad_token_id = model.prior_tokenizer.encode(
                    '*',
                    add_special_tokens=False,
                    max_length=token_count,
                )[0]
                prior_text_encoder_initial_token_ids += [prior_text_encoder_pad_token_id] * (
                        token_count - len(prior_text_encoder_initial_token_ids))
                for token_id, initial_token_id in zip(prior_text_encoder_token_ids, prior_text_encoder_initial_token_ids):
                    all_prior_text_encoder_token_embeds[token_id] = all_prior_text_encoder_token_embeds[initial_token_id]

                model.embeddings = [
                    StableDiffusionModelEmbedding(
                        "*", all_prior_text_encoder_token_embeds[self.prior_text_encoder_trainable_token_embeds_mask],
                        token_count)]

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

        model.prior_text_encoder.train()
        model.prior_prior.train()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        # reset untrainable embeddings
        with torch.no_grad():
            model.prior_text_encoder.get_input_embeddings().weight[
                self.prior_text_encoder_untrainable_token_embeds_mask
            ] = self.all_prior_text_encoder_original_token_embeds[self.prior_text_encoder_untrainable_token_embeds_mask]

        # save back to model
        model.embeddings = [WuerstchenModelEmbedding(
            "*",
            model.prior_text_encoder.get_input_embeddings().weight[self.prior_text_encoder_trainable_token_embeds_mask],
            model.embeddings[0].token_count
        )]
