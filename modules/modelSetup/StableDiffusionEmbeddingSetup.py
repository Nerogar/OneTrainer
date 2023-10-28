from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionEmbeddingSetup(BaseStableDiffusionSetup):
    all_token_embeds: Tensor
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
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        return model.text_encoder.get_input_embeddings().parameters()

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            {
                'params': model.text_encoder.get_input_embeddings().parameters(),
                'lr': args.learning_rate,
                'initial_lr': args.learning_rate,
            }
        ]

    def setup_model(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.requires_grad_(False)
        model.text_encoder.get_input_embeddings().requires_grad_(True)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        token_count = args.token_count if len(model.embeddings) == 0 else model.embeddings[0].token_count

        tokens = [f"<embedding_{i}>" for i in range(token_count)]
        model.tokenizer.add_tokens(tokens)
        model.text_encoder.resize_token_embeddings(len(model.tokenizer))

        model.text_encoder.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())

        with torch.no_grad():
            token_ids = model.tokenizer.encode(
                tokens,
                add_special_tokens=False,
            )

            self.all_token_embeds = model.text_encoder.get_input_embeddings().weight.data
            self.all_original_token_embeds = self.all_token_embeds.clone()
            self.trainable_token_embeds_mask = [(i in token_ids) for i in range(len(self.all_original_token_embeds))]
            self.untrainable_token_embeds_mask = [
                (i not in token_ids) for i in range(len(self.all_original_token_embeds))
            ]

            if len(model.embeddings) > 0:
                # an embedding was loaded
                for i, token_id in enumerate(token_ids):
                    self.all_token_embeds[token_id] = model.embeddings[0].vector[i]
            else:
                # create a new embedding
                initial_token_ids = model.tokenizer.encode(
                    args.initial_embedding_text,
                    add_special_tokens=False,
                    max_length=token_count,
                )
                pad_token_id = model.tokenizer.encode(
                    '*',
                    add_special_tokens=False,
                    max_length=token_count,
                )[0]
                initial_token_ids += [pad_token_id] * (token_count - len(initial_token_ids))
                for token_id, initial_token_id in zip(token_ids, initial_token_ids):
                    self.all_token_embeds[token_id] = self.all_token_embeds[initial_token_id]

                model.embeddings = [
                    StableDiffusionModelEmbedding(
                        "*", self.all_token_embeds[self.trainable_token_embeds_mask],
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
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        vae_on_train_device = self.debug_mode or args.align_prop_loss

        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet.to(self.train_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        model.text_encoder.train()
        model.vae.eval()
        model.unet.train()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        # reset untrainable embeddings
        with torch.no_grad():
            model.text_encoder.get_input_embeddings().weight[
                self.untrainable_token_embeds_mask
            ] = self.all_original_token_embeds[self.untrainable_token_embeds_mask]

        # save back to model
        model.embeddings = [StableDiffusionModelEmbedding(
            "*", self.all_token_embeds[self.trainable_token_embeds_mask], model.embeddings[0].token_count
        )]
