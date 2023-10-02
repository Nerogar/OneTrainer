from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenFineTuneSetup(BaseWuerstchenSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(WuerstchenFineTuneSetup, self).__init__(
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

        params += list(model.prior_prior.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.train_unet:
            lr = args.unet_learning_rate if args.unet_learning_rate is not None else args.learning_rate
            param_groups.append({
                'params': model.prior_prior.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        return param_groups

    def setup_model(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.prior_text_encoder.requires_grad_(train_text_encoder)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.prior_prior.requires_grad_(train_unet)

        model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

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
            model: WuerstchenModel
    ):
        model.decoder_text_encoder.to(self.train_device)
        model.decoder_decoder.to(self.train_device)
        model.decoder_vqgan.to(self.train_device)
        model.effnet_encoder.to(self.train_device)
        model.prior_text_encoder.to(self.train_device)
        model.prior_prior.to(self.train_device)

        model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()
        model.prior_text_encoder.eval()
        model.prior_prior.eval()

    def setup_train_device(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        model.decoder_text_encoder.to(self.temp_device)
        model.decoder_decoder.to(self.temp_device)
        model.decoder_vqgan.to(self.temp_device)
        model.effnet_encoder.to(self.temp_device)
        model.prior_text_encoder.to(self.train_device)
        model.prior_prior.to(self.train_device)

        if args.train_text_encoder:
            model.prior_text_encoder.train()
        else:
            model.prior_text_encoder.eval()

        model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        if args.train_unet:
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