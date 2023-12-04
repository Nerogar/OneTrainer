from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LearningRateScaler import LearningRateScaler


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
        batch_size = 1 if args.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.GRADIENT_ACCUMULATION] else args.batch_size
        gradient_accumulation_steps = 1 if args.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.BATCH] else args.gradient_accumulation_steps

        if args.train_prior:
            lr = args.prior_learning_rate if args.prior_learning_rate is not None else args.learning_rate
            lr = lr * ((batch_size * gradient_accumulation_steps) ** 0.5)

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

        train_prior = args.train_prior and (model.train_progress.epoch < args.train_prior_epochs)
        model.prior_prior.requires_grad_(train_prior)

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