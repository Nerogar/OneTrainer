from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModelEmbedding
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLEmbeddingSetup(BaseStableDiffusionXLSetup):
    all_text_encoder_1_token_embeds: Tensor
    all_text_encoder_1_original_token_embeds: Tensor
    text_encoder_1_trainable_token_embeds_mask: list[bool]
    text_encoder_1_untrainable_token_embeds_mask: list[bool]

    all_text_encoder_2_token_embeds: Tensor
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
            {
                'params': model.text_encoder_1.get_input_embeddings().parameters(),
                'lr': args.learning_rate,
                'initial_lr': args.learning_rate,
            },
            {
                'params': model.text_encoder_2.get_input_embeddings().parameters(),
                'lr': args.learning_rate,
                'initial_lr': args.learning_rate,
            }
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

        token_count = args.token_count if len(model.embeddings) == 0 else model.embeddings[0].token_count

        tokens = [f"<embedding_{i}>" for i in range(token_count)]
        model.tokenizer_1.add_tokens(tokens)
        model.tokenizer_2.add_tokens(tokens)
        model.text_encoder_1.resize_token_embeddings(len(model.tokenizer_1))
        model.text_encoder_2.resize_token_embeddings(len(model.tokenizer_2))

        with torch.no_grad():
            text_encoder_1_token_ids = model.tokenizer_1.encode(
                tokens,
                add_special_tokens=False,
            )

            self.all_text_encoder_1_token_embeds = model.text_encoder_1.get_input_embeddings().weight.data
            self.all_text_encoder_1_original_token_embeds = self.all_text_encoder_1_token_embeds.clone()
            self.text_encoder_1_trainable_token_embeds_mask = \
                [(i in text_encoder_1_token_ids) for i in range(len(self.all_text_encoder_1_original_token_embeds))]
            self.text_encoder_1_untrainable_token_embeds_mask = [
                (i not in text_encoder_1_token_ids) for i in range(len(self.all_text_encoder_1_original_token_embeds))
            ]

            text_encoder_2_token_ids = model.tokenizer_1.encode(
                tokens,
                add_special_tokens=False,
            )

            self.all_text_encoder_2_token_embeds = model.text_encoder_2.get_input_embeddings().weight.data
            self.all_text_encoder_2_original_token_embeds = self.all_text_encoder_2_token_embeds.clone()
            self.text_encoder_2_trainable_token_embeds_mask = \
                [(i in text_encoder_2_token_ids) for i in range(len(self.all_text_encoder_2_original_token_embeds))]
            self.text_encoder_2_untrainable_token_embeds_mask = [
                (i not in text_encoder_2_token_ids) for i in range(len(self.all_text_encoder_2_original_token_embeds))
            ]

            if len(model.embeddings) > 0:
                # an embedding was loaded
                for i, token_id in enumerate(text_encoder_1_token_ids):
                    self.all_text_encoder_1_token_embeds[token_id] = model.embeddings[0].text_encoder_1_vector[i]
                for i, token_id in enumerate(text_encoder_2_token_ids):
                    self.all_text_encoder_2_token_embeds[token_id] = model.embeddings[0].text_encoder_2_vector[i]
            else:
                # create a new embedding
                text_encoder_1_initial_token_ids = model.tokenizer_1.encode(
                    args.initial_embedding_text,
                    add_special_tokens=False,
                    max_length=token_count,
                )
                text_encoder_1_pad_token_id = model.tokenizer_1.encode(
                    '*',
                    add_special_tokens=False,
                    max_length=token_count,
                )[0]
                text_encoder_1_initial_token_ids += [text_encoder_1_pad_token_id] * (
                        token_count - len(text_encoder_1_initial_token_ids))
                for token_id, initial_token_id in zip(text_encoder_1_token_ids, text_encoder_1_initial_token_ids):
                    self.all_text_encoder_1_token_embeds[token_id] = self.all_text_encoder_1_token_embeds[
                        initial_token_id]

                model.embeddings = [
                    StableDiffusionModelEmbedding(
                        "*", self.all_text_encoder_1_token_embeds[self.text_encoder_1_trainable_token_embeds_mask],
                        token_count)]

                text_encoder_2_initial_token_ids = model.tokenizer_2.encode(
                    args.initial_embedding_text,
                    add_special_tokens=False,
                    max_length=token_count,
                )
                text_encoder_2_pad_token_id = model.tokenizer_2.encode(
                    '*',
                    add_special_tokens=False,
                    max_length=token_count,
                )[0]
                text_encoder_2_initial_token_ids += [text_encoder_2_pad_token_id] * (
                        token_count - len(text_encoder_2_initial_token_ids))
                for token_id, initial_token_id in zip(text_encoder_2_token_ids, text_encoder_2_initial_token_ids):
                    self.all_text_encoder_2_token_embeds[token_id] = self.all_text_encoder_2_token_embeds[
                        initial_token_id]

                model.embeddings = [
                    StableDiffusionModelEmbedding(
                        "*", self.all_text_encoder_2_token_embeds[self.text_encoder_2_trainable_token_embeds_mask],
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

    def setup_eval_device(
            self,
            model: StableDiffusionXLModel
    ):
        model.text_encoder_1.to(self.train_device)
        model.text_encoder_2.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)

        model.text_encoder_1.eval()
        model.text_encoder_2.eval()
        model.vae.eval()
        model.unet.eval()

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        model.text_encoder_1.to(self.train_device)
        model.text_encoder_2.to(self.train_device)
        model.vae.to(self.temp_device)
        model.unet.to(self.train_device)

        model.text_encoder_1.train()
        model.text_encoder_2.train()
        model.vae.eval()
        model.unet.train()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        # reset untrainable embeddings
        with torch.no_grad():
            model.text_encoder_1.get_input_embeddings().weight[
                self.text_encoder_1_untrainable_token_embeds_mask
            ] = self.all_text_encoder_1_original_token_embeds[self.text_encoder_1_untrainable_token_embeds_mask]

            model.text_encoder_2.get_input_embeddings().weight[
                self.text_encoder_2_untrainable_token_embeds_mask
            ] = self.all_text_encoder_2_original_token_embeds[self.text_encoder_2_untrainable_token_embeds_mask]

        # save back to model
        model.embeddings = [StableDiffusionXLModelEmbedding(
            "*",
            self.all_text_encoder_1_token_embeds[self.text_encoder_1_trainable_token_embeds_mask],
            self.all_text_encoder_2_token_embeds[self.text_encoder_2_trainable_token_embeds_mask],
            model.embeddings[0].token_count
        )]
