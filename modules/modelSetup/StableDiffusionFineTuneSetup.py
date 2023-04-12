import torch
from diffusers.utils.import_utils import is_xformers_available

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionFineTuneSetup(BaseModelSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def setup_gradients(
            self,
            model: StableDiffusionModel,
            epoch: int,
            args: TrainArgs,
    ):
        train_text_encoder = args.train_text_encoder and (epoch < args.train_text_encoder_epochs)

        model.text_encoder.requires_grad_(train_text_encoder)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(True)

    def setup_eval_device(
            self,
            model: StableDiffusionModel
    ):
        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.train_device)

        model.vae.eval()
        model.unet.eval()

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device if self.debug_mode else self.temp_device)
        model.unet.to(self.train_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        if is_xformers_available():
            try:
                model.vae.enable_xformers_memory_efficient_attention()
                model.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        model.unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            model.text_encoder.gradient_checkpointing_enable()

        model.vae.train()
        model.unet.train()
