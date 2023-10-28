from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.KandinskyModel import KandinskyModel
from modules.modelSetup.BaseKandinskySetup import BaseKandinskySetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class KandinskyFineTuneSetup(BaseKandinskySetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(KandinskyFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: KandinskyModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        # params += list(model.prior_text_encoder.parameters())
        # params += list(model.prior_image_encoder.parameters())
        # params += list(model.prior_prior.parameters())

        # params += list(model.text_encoder.parameters())
        params += list(model.unet.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: KandinskyModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return self.create_parameters(model, args)

    def setup_model(
            self,
            model: KandinskyModel,
            args: TrainArgs,
    ):
        model.prior_text_encoder.requires_grad_(False)
        model.prior_image_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)

        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(True)
        model.movq.requires_grad_(False)

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
            model: KandinskyModel,
            args: TrainArgs,
    ):
        model.prior_text_encoder.to(self.temp_device)
        model.prior_image_encoder.to(self.temp_device)
        model.prior_prior.to(self.temp_device)

        model.text_encoder.to(self.train_device)
        model.unet.to(self.train_device)
        model.movq.to(self.train_device if self.debug_mode else self.temp_device)

        model.prior_text_encoder.train()
        model.prior_image_encoder.train()
        model.prior_prior.train()

        model.text_encoder.train()
        model.unet.train()
        model.movq.train()

    def after_optimizer_step(
            self,
            model: KandinskyModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.text_encoder.requires_grad_(train_text_encoder)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet.requires_grad_(train_unet)
