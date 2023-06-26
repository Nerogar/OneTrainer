from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.KandinskyModel import KandinskyModel
from modules.modelSetup.BaseKandinskySetup import BaseKandinskySetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class KandinskyLoRASetup(BaseKandinskySetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(KandinskyLoRASetup, self).__init__(
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

        if args.train_unet:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: KandinskyModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

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
            model: KandinskyModel,
            args: TrainArgs,
    ):
        if model.unet_lora is None:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, args.lora_rank, "lora_unet", args.lora_alpha, ["attentions"]
            )

        model.prior_text_encoder.requires_grad_(False)
        model.prior_image_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)

        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.movq.requires_grad_(False)

        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet_lora.requires_grad_(train_unet)

        model.unet_lora.hook_to_module()

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, args), model.optimizer_state_dict, args
        )
        del model.optimizer_state_dict

        self.setup_optimizations(model, args)

    def setup_eval_device(
            self,
            model: KandinskyModel
    ):
        model.prior_text_encoder.to(self.train_device)
        model.prior_image_encoder.to(self.train_device)
        model.prior_prior.to(self.train_device)

        model.text_encoder.to(self.train_device)
        model.unet.to(self.train_device)
        model.movq.to(self.train_device)

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

        model.prior_text_encoder.eval()
        model.prior_image_encoder.eval()
        model.prior_prior.eval()

        model.text_encoder.eval()
        model.unet.eval()
        model.movq.eval()

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

        if model.unet_lora is not None:
            model.unet_lora.to(self.train_device)

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
        train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
        model.unet_lora.requires_grad_(train_unet)
