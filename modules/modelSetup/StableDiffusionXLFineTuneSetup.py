from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLFineTuneSetup(BaseStableDiffusionXLSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLFineTuneSetup, self).__init__(
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

        if args.train_text_encoder:
            params += list(model.text_encoder_1.parameters())

        if args.train_text_encoder_2:
            params += list(model.text_encoder_2.parameters())

        if args.train_unet:
            params += list(model.unet.parameters())

        return params
        

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.train_text_encoder:
            self.create_param_groups(args, model.text_encoder_1.parameters(), args.text_encoder_learning_rate, param_groups)
            
        if args.train_text_encoder_2:
            self.create_param_groups(args, model.text_encoder_2.parameters(), args.text_encoder_2_learning_rate, param_groups)

        if args.train_unet:
            self.create_param_groups(args, model.unet.parameters(), args.unet_learning_rate, param_groups)

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        train_text_encoder_1 = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.text_encoder_1.requires_grad_(train_text_encoder_1)

        train_text_encoder_2 = args.train_text_encoder_2 and (model.train_progress.epoch < args.train_text_encoder_2_epochs)
        model.text_encoder_2.requires_grad_(train_text_encoder_2)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet.requires_grad_(train_unet)

        model.vae.requires_grad_(False)

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
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        vae_on_train_device = args.align_prop
        text_encoder_1_on_train_device = args.train_text_encoder or args.align_prop or not args.latent_caching
        text_encoder_2_on_train_device = args.train_text_encoder_2 or args.align_prop or not args.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if args.train_text_encoder:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if args.train_text_encoder_2:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.train()

        if args.train_unet:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        train_text_encoder_1 = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        model.text_encoder_1.requires_grad_(train_text_encoder_1)

        train_text_encoder_2 = args.train_text_encoder_2 and (model.train_progress.epoch < args.train_text_encoder_2_epochs)
        model.text_encoder_2.requires_grad_(train_text_encoder_2)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet.requires_grad_(train_unet)
