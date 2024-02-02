from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSetup.BasePixArtAlphaSetup import BasePixArtAlphaSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class PixArtAlphaFineTuneSetup(BasePixArtAlphaSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(PixArtAlphaFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        params = list()

        if config.train_text_encoder:
            params += list(model.text_encoder.parameters())

        if config.train_prior:
            params += list(model.transformer.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if config.train_text_encoder:
            param_groups.append(
                self.create_param_groups(config, model.text_encoder.parameters(), config.text_encoder_learning_rate)
            )

        if config.train_prior:
            param_groups.append(
                self.create_param_groups(config, model.transformer.parameters(), config.prior_learning_rate)
            )

        return param_groups

    def setup_model(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        train_text_encoder = config.train_text_encoder and (model.train_progress.epoch < config.train_text_encoder_epochs)
        model.text_encoder.requires_grad_(train_text_encoder)

        train_prior = config.train_prior and (model.train_progress.epoch < config.train_prior_epochs)
        model.transformer.requires_grad_(train_prior)

        model.vae.requires_grad_(False)

        # if args.rescale_noise_scheduler_to_zero_terminal_snr:
        #     model.rescale_noise_scheduler_to_zero_terminal_snr()
        #     model.force_v_prediction()
        # elif args.force_v_prediction:
        #     model.force_v_prediction()
        # elif args.force_epsilon_prediction:
        #     model.force_epsilon_prediction()

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, config), model.optimizer_state_dict, config
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, config), model.ema_state_dict, config
        )
        del model.ema_state_dict

        self.setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop
        text_encoder_on_train_device = config.train_text_encoder or config.align_prop or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if config.train_text_encoder:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if config.train_prior:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        train_text_encoder = config.train_text_encoder and (model.train_progress.epoch < config.train_text_encoder_epochs)
        model.text_encoder.requires_grad_(train_text_encoder)

        train_prior = config.train_prior and (model.train_progress.epoch < config.train_prior_epochs)
        model.transformer.requires_grad_(train_prior)
