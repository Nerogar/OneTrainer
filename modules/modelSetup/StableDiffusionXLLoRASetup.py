from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionXLLoRASetup(BaseStableDiffusionXLSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        params = list()

        if config.text_encoder.train:
            params += list(model.text_encoder_1_lora.parameters())

        if config.text_encoder_2.train:
            params += list(model.text_encoder_2_lora.parameters())

        if config.unet.train:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if config.text_encoder.train:
            param_groups.append(
                self.create_param_groups(config, model.text_encoder_1_lora.parameters(),
                                         config.text_encoder.learning_rate)
            )

        if config.text_encoder_2.train:
            param_groups.append(
                self.create_param_groups(
                    config, model.text_encoder_2_lora.parameters(), config.text_encoder_2.learning_rate
                )
            )

        if config.unet.train:
            param_groups.append(
                self.create_param_groups(config, model.unet_lora.parameters(), config.unet.learning_rate)
            )

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if model.text_encoder_1_lora is None and config.text_encoder.train:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, config.lora_rank, "lora_te1", config.lora_alpha
            )

        if model.text_encoder_2_lora is None and config.text_encoder_2.train:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, config.lora_rank, "lora_te2", config.lora_alpha
            )

        if model.unet_lora is None and config.unet.train:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, config.lora_rank, "lora_unet", config.lora_alpha, ["attentions"]
            )

        if model.text_encoder_1_lora:
            model.text_encoder_1_lora.set_dropout(config.dropout_probability)
        if model.text_encoder_2_lora:
            model.text_encoder_2_lora.set_dropout(config.dropout_probability)
        if model.unet_lora:
            model.unet_lora.set_dropout(config.dropout_probability)

        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)
        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)
        if model.unet_lora is not None:
            train_unet = config.unet.train and \
                         not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.hook_to_module()
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.hook_to_module()
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        if model.unet_lora is not None:
            model.unet_lora.hook_to_module()
            model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())

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
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop
        text_encoder_1_on_train_device = config.text_encoder.train or config.align_prop or not config.latent_caching
        text_encoder_2_on_train_device = config.text_encoder_2.train or config.align_prop or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.unet_lora is not None:
            train_unet = config.unet.train and \
                         not self.stop_unet_training_elapsed(config, model.train_progress)
            model.unet_lora.requires_grad_(train_unet)

    def report_learning_rates(
            self,
            model,
            config,
            scheduler,
            tensorboard
    ):
        lrs = scheduler.get_last_lr()
        names = []
        if config.text_encoder.train:
            names.append("te")
        if config.text_encoder_2.train:
            names.append("te2")
        if config.unet.train:
            names.append("unet")
        assert len(lrs) == len(names)

        lrs = config.optimizer.optimizer.maybe_adjust_lrs(lrs, model.optimizer)

        for name, lr in zip(names, lrs):
            tensorboard.add_scalar(
                f"lr/{name}", lr, model.train_progress.global_step
            )
