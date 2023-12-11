from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenLoRASetup(BaseWuerstchenSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(WuerstchenLoRASetup, self).__init__(
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

        if args.train_text_encoder:
            params += list(model.prior_text_encoder_lora.parameters())

        if args.train_prior:
            params += list(model.prior_prior_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()
        
        if args.train_text_encoder:
            self.create_param_groups(args, model.prior_text_encoder_lora.parameters(), args.text_encoder_learning_rate, param_groups)

        if args.prior:
            self.create_param_groups(args, model.prior_prior_lora.parameters(), args.prior_learning_rate, param_groups)

        return param_groups

    def setup_model(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        if model.prior_text_encoder_lora is None and args.train_text_encoder:
            model.prior_text_encoder_lora = LoRAModuleWrapper(
                model.prior_text_encoder, args.lora_rank, "lora_prior_te", args.lora_alpha
            )

        if model.prior_prior_lora is None and args.train_prior:
            model.prior_prior_lora = LoRAModuleWrapper(
                model.prior_prior, args.lora_rank, "lora_prior_prior", args.lora_alpha, ["attention"]
            )

        model.prior_text_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)
        model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        if model.prior_text_encoder_lora is not None:
            model.prior_text_encoder_lora.requires_grad_(train_text_encoder)

        train_prior = args.train_prior and (model.train_progress.epoch < args.train_prior_epochs)
        if model.prior_prior_lora is not None:
            model.prior_prior_lora.requires_grad_(train_prior)

        if model.prior_text_encoder_lora is not None:
            model.prior_text_encoder_lora.hook_to_module()
            model.prior_text_encoder_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.prior_prior_lora is not None:
            model.prior_prior_lora.hook_to_module()
            model.prior_prior_lora.to(dtype=args.lora_weight_dtype.torch_dtype())

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

        text_encoder_on_train_device = args.train_text_encoder or args.align_prop or not args.latent_caching

        model.prior_text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.prior_prior_to(self.train_device)

        model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        if args.train_text_encoder:
            model.prior_text_encoder.train()
        else:
            model.prior_text_encoder.eval()

        if args.train_prior:
            model.prior_prior.train()
        else:
            model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        pass
