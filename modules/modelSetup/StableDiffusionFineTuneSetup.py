from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.modelSetup.mixin.ModelSetupClipEmbeddingMixin import ModelSetupClipEmbeddingMixin
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionFineTuneSetup(
    BaseStableDiffusionSetup,
    ModelSetupClipEmbeddingMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        if args.train_text_encoder:
            params += list(model.text_encoder.parameters())

            if args.train_embedding:
                params += list(model.text_encoder.get_input_embeddings().parameters())

        if args.train_embedding and not args.train_text_encoder:
            params += list(model.text_encoder.get_input_embeddings().parameters())

        if args.train_unet:
            params += list(model.unet.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()
        
        if args.train_text_encoder:
            text_encoder_parameters = list(model.text_encoder.parameters())
            if args.train_embedding:
                for embedding_parameter in model.text_encoder.get_input_embeddings().parameters():
                    text_encoder_parameters.remove(embedding_parameter)

            param_groups.append(
                self.create_param_groups(args, model.text_encoder.parameters(), args.text_encoder_learning_rate)
            )

        if args.train_embedding and not args.train_text_encoder:
            param_groups.append(
                self.create_param_groups(
                    args,
                    model.text_encoder.get_input_embeddings().parameters(),
                    args.embedding_learning_rate,
                )
            )

        if args.train_unet:
            param_groups.append(
                self.create_param_groups(args, model.unet.parameters(), args.unet_learning_rate)
            )

        return param_groups

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.text_encoder.requires_grad_(train_text_encoder)

        train_embedding = args.train_embedding and (model.train_progress.epoch < args.train_embedding_epochs)
        if train_embedding:
            model.text_encoder.get_input_embeddings().requires_grad_(True)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet.requires_grad_(train_unet)

        model.vae.requires_grad_(False)


    def setup_model(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        self.__setup_requires_grad(model, args)

        if args.train_embedding:
            model.text_encoder.get_input_embeddings().to(dtype=args.embedding_weight_dtype.torch_dtype())

        if args.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()
        elif args.force_v_prediction:
            model.force_v_prediction()
        elif args.force_epsilon_prediction:
            model.force_epsilon_prediction()

        if len(model.embeddings) == 0:
            vector = self._create_new_embedding(
                model.tokenizer,
                model.text_encoder,
                args.initial_embedding_text,
                args.token_count,
            )

            model.embeddings = [StableDiffusionModelEmbedding(vector, 'embedding')]

        original_token_embeds, untrainable_token_ids = self._add_embeddings_to_clip(
            model.tokenizer,
            model.text_encoder,
            [(model.embeddings[0].text_encoder_vector, model.embeddings[0].text_tokens, True)],
        )
        model.all_text_encoder_original_token_embeds = original_token_embeds
        model.text_encoder_untrainable_token_embeds_mask = untrainable_token_ids

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
        vae_on_train_device = self.debug_mode or args.align_prop
        text_encoder_on_train_device = \
            args.train_text_encoder \
            or args.train_embedding \
            or args.align_prop \
            or not args.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        if args.train_text_encoder:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if args.train_unet:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, args)

        self._embeddigns_after_optimizer_step(
            model.text_encoder.get_input_embeddings(),
            model.all_text_encoder_original_token_embeds,
            model.text_encoder_untrainable_token_embeds_mask,
        )
