from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


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
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()

        if args.train_text_encoder:
            params += list(model.text_encoder_1_lora.parameters())
            params += list(model.text_encoder_2_lora.parameters())

        if args.train_unet:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.train_text_encoder:
            lr = args.text_encoder_learning_rate if args.text_encoder_learning_rate is not None else args.learning_rate
            param_groups.append({
                'params': model.text_encoder_1_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })
            param_groups.append({
                'params': model.text_encoder_2_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        if args.train_unet:
            lr = args.unet_learning_rate if args.unet_learning_rate is not None else args.learning_rate
            param_groups.append({
                'params': model.unet_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        if model.text_encoder_1_lora is None and args.train_text_encoder:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, args.lora_rank, "lora_te1", args.lora_alpha
            )

        if model.text_encoder_2_lora is None and args.train_text_encoder:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, args.lora_rank, "lora_te2", args.lora_alpha
            )

        if model.unet_lora is None and args.train_unet:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, args.lora_rank, "lora_unet", args.lora_alpha, ["attentions"]
            )

        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.requires_grad_(train_text_encoder)
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.requires_grad_(train_text_encoder)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        if model.unet_lora is not None:
            model.unet_lora.requires_grad_(train_unet)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.hook_to_module()
            model.text_encoder_1_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.hook_to_module()
            model.text_encoder_2_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.unet_lora is not None:
            model.unet_lora.hook_to_module()
            model.unet_lora.to(dtype=args.lora_weight_dtype.torch_dtype())

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
            model: StableDiffusionXLModel
    ):
        model.text_encoder_1.to(self.train_device)
        model.text_encoder_2.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.to(self.train_device)

        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.to(self.train_device)

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

        model.text_encoder_1.eval()
        model.text_encoder_2.eval()
        model.vae.eval()
        model.unet.eval()

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        model.text_encoder_1.to(self.train_device if args.train_text_encoder else self.temp_device)
        model.text_encoder_2.to(self.train_device if args.train_text_encoder else self.temp_device)
        model.vae.to(self.temp_device)
        model.unet.to(self.train_device)

        if model.text_encoder_1_lora is not None and args.train_text_encoder:
            model.text_encoder_1_lora.to(self.train_device)

        if model.text_encoder_2_lora is not None and args.train_text_encoder:
            model.text_encoder_2_lora.to(self.train_device)

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

        if args.train_text_encoder:
            model.text_encoder_1.train()
            model.text_encoder_2.train()
        else:
            model.text_encoder_1.eval()
            model.text_encoder_2.eval()
        model.vae.eval()
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
        train_text_encoder = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.requires_grad_(train_text_encoder)
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.requires_grad_(train_text_encoder)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        if model.unet_lora is not None:
            model.unet_lora.requires_grad_(train_unet)
