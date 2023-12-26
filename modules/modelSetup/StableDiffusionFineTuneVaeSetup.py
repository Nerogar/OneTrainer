from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionFineTuneVaeSetup(BaseStableDiffusionSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionFineTuneVaeSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        return model.vae.decoder.parameters()

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            self.create_param_groups(
                args,
                model.vae.decoder.parameters(),
                args.learning_rate,
            )
        ]

    def setup_model(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.vae.decoder.requires_grad_(True)
        model.unet.requires_grad_(False)

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
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.to(self.temp_device)
        model.vae.to(self.train_device)
        model.unet.to(self.temp_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        model.text_encoder.eval()
        model.vae.train()
        model.unet.eval()

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        latent_image = batch['latent_image']
        image = batch['image']

        predicted_image = model.vae.decode(latent_image, return_dict=True).sample

        model_output_data = {
            'loss_type': 'target',
            'predicted': predicted_image,
            'target': image,
        }

        if args.debug_mode:
            with torch.no_grad():
                # image
                self._save_image(image, args.debug_dir + "/training_batches", "1-image", train_progress.global_step)

                # predicted image
                predicted_image_clamped = predicted_image.clamp(-1, 1)
                self._save_image(
                    predicted_image_clamped, args.debug_dir + "/training_batches", "2-predicted_image",
                    train_progress.global_step
                )

        return model_output_data

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        pass
