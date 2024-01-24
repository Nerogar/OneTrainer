from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSetup.BasePixArtAlphaSetup import BasePixArtAlphaSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class PixArtAlphaLoRASetup(BasePixArtAlphaSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(PixArtAlphaLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        if args.train_text_encoder:
            params += list(model.text_encoder_lora.parameters())

        if args.train_prior:
            params += list(model.transformer_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.train_text_encoder:
            param_groups.append(
                self.create_param_groups(args, model.text_encoder_lora.parameters(), args.text_encoder_learning_rate)
            )

        if args.train_prior:
            param_groups.append(
                self.create_param_groups(args, model.transformer_lora.parameters(), args.prior_learning_rate)
            )

        return param_groups

    def setup_model(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
    ):
        if model.text_encoder_lora is None and args.train_text_encoder:
            model.text_encoder_lora = LoRAModuleWrapper(
                model.text_encoder, args.lora_rank, "lora_te", args.lora_alpha
            )

        if model.transformer_lora is None and args.train_prior:
            model.transformer_lora = LoRAModuleWrapper(
                model.transformer, args.lora_rank, "lora_transformer", args.lora_alpha, ["attn1", "attn2"]
            )

        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        if model.text_encoder_lora is not None:
            model.text_encoder_lora.requires_grad_(train_text_encoder)

        train_prior = args.train_prior and (model.train_progress.epoch < args.train_prior_epochs)
        if model.transformer_lora is not None:
            model.transformer_lora.requires_grad_(train_prior)

        model.vae.requires_grad_(False)

        # if args.rescale_noise_scheduler_to_zero_terminal_snr:
        #     model.rescale_noise_scheduler_to_zero_terminal_snr()
        #     model.force_v_prediction()
        # elif args.force_v_prediction:
        #     model.force_v_prediction()
        # elif args.force_epsilon_prediction:
        #     model.force_epsilon_prediction()

        if model.text_encoder_lora is not None:
            model.text_encoder_lora.hook_to_module()
            model.text_encoder_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.transformer_lora is not None:
            model.transformer_lora.hook_to_module()
            model.transformer_lora.to(dtype=args.lora_weight_dtype.torch_dtype())

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
            model: PixArtAlphaModel,
            args: TrainArgs,
    ):
        vae_on_train_device = self.debug_mode or args.align_prop
        text_encoder_on_train_device = args.train_text_encoder or args.align_prop or not args.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if args.train_text_encoder:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if args.train_prior:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.text_encoder.requires_grad_(train_text_encoder)

        train_prior = args.train_prior and (model.train_progress.epoch < args.train_prior_epochs)
        model.transformer.requires_grad_(train_prior)
