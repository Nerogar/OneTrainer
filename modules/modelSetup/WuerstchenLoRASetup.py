from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


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
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        params = list()

        if config.train_text_encoder:
            params += list(model.prior_text_encoder_lora.parameters())

        if config.train_prior:
            params += list(model.prior_prior_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()
        
        if config.train_text_encoder:
            param_groups.append(
                self.create_param_groups(
                    config, model.prior_text_encoder_lora.parameters(), config.text_encoder_learning_rate
                )
            )

        if config.train_prior:
            param_groups.append(
                self.create_param_groups(config, model.prior_prior_lora.parameters(), config.prior_learning_rate)
            )

        return param_groups

    def setup_model(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        if model.prior_text_encoder_lora is None and config.train_text_encoder:
            model.prior_text_encoder_lora = LoRAModuleWrapper(
                model.prior_text_encoder, config.lora_rank, "lora_prior_te", config.lora_alpha, type=config.lora_type
            )

        if model.prior_prior_lora is None and config.train_prior:
            model.prior_prior_lora = LoRAModuleWrapper(
                model.prior_prior, config.lora_rank, "lora_prior_unet", config.lora_alpha, ["attention"], config.lora_type
            )

        model.prior_text_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)
        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

        train_text_encoder = config.train_text_encoder and (model.train_progress.epoch < config.train_text_encoder_epochs)
        if model.prior_text_encoder_lora is not None:
            model.prior_text_encoder_lora.requires_grad_(train_text_encoder)

        train_prior = config.train_prior and (model.train_progress.epoch < config.train_prior_epochs)
        if model.prior_prior_lora is not None:
            model.prior_prior_lora.requires_grad_(train_prior)

        if model.prior_text_encoder_lora is not None:
            model.prior_text_encoder_lora.hook_to_module()
            model.prior_text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        if model.prior_prior_lora is not None:
            model.prior_prior_lora.hook_to_module()
            model.prior_prior_lora.to(dtype=config.lora_weight_dtype.torch_dtype())

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
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder_to(self.temp_device)
        model.decoder_decoder_to(self.temp_device)
        model.decoder_vqgan_to(self.temp_device)
        model.effnet_encoder_to(self.temp_device)

        text_encoder_on_train_device = config.train_text_encoder or config.align_prop or not config.latent_caching

        model.prior_text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.prior_prior_to(self.train_device)

        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        if config.train_text_encoder:
            model.prior_text_encoder.train()
        else:
            model.prior_text_encoder.eval()

        if config.train_prior:
            model.prior_prior.train()
        else:
            model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        pass
