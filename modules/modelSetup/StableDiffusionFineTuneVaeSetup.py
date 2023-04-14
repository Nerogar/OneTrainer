from typing import Iterator

import torch
from diffusers.utils.import_utils import is_xformers_available
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionFineTuneVaeSetup(BaseModelSetup):
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
    ) -> Iterator[Parameter]:
        return model.vae.decoder.parameters()

    def setup_model(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.vae.decoder.requires_grad_(True)
        model.unet.requires_grad_(False)

        if model.optimizer_state_dict is not None and model.optimizer is None:
            model.optimizer = create.create_optimizer(self.create_parameters(model, args), args)
            # TODO: this will break if the optimizer class changed during a restart
            model.optimizer.load_state_dict(model.optimizer_state_dict)
            del model.optimizer_state_dict
        elif model.optimizer_state_dict is None and model.optimizer is None:
            model.optimizer = create.create_optimizer(self.create_parameters(model, args), args)

    def setup_eval_device(
            self,
            model: StableDiffusionModel
    ):
        model.text_encoder.to(self.temp_device)
        model.vae.to(self.train_device)
        model.unet.to(self.temp_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        model.text_encoder.eval()
        model.vae.eval()
        model.unet.eval()

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

        if is_xformers_available():
            try:
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        model.vae.enable_gradient_checkpointing()

        model.text_encoder.eval()
        model.vae.train()
        model.unet.eval()

    def create_optimizer(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Optimizer:
        return model.optimizer

    def get_train_progress(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> TrainProgress:
        return model.train_progress

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress
    ) -> (Tensor, Tensor):
        latent_image = batch['latent_image']
        image = batch['image']

        predicted_image = model.vae.decode(latent_image, return_dict=True).sample

        if args.debug_mode:
            with torch.no_grad():
                # image
                self.save_image(image, args.debug_dir + "/training_batches", "1-image", train_progress.global_step)

                # predicted image
                predicted_image_clamped = predicted_image.clamp(-1, 1)
                self.save_image(predicted_image_clamped, args.debug_dir + "/training_batches", "2-predicted_image", train_progress.global_step)

        return predicted_image, image
