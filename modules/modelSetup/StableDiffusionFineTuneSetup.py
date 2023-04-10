import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.cross_attention import CrossAttnProcessor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup


class StableDiffusionFineTuneSetup(BaseModelSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,

    ):
        super(StableDiffusionFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
        )

    def start_data_loader(
            self,
            model: StableDiffusionModel,
    ):
        model.text_encoder.to(self.temp_device)
        model.vae.to(self.train_device)
        model.unet.to(self.temp_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.train_device)

        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.vae.eval()
        model.unet.eval()

    def start_train(
            self,
            model: StableDiffusionModel,
            train_text_encoder: bool,
    ):
        model.text_encoder.to(self.train_device)
        model.vae.to(self.temp_device)
        model.unet.to(self.train_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.temp_device)

        if False:
            model.unet.set_attn_processor(CrossAttnProcessor())
        else:
            model.unet.set_attn_processor(AttnProcessor2_0())

        model.text_encoder.requires_grad_(train_text_encoder)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(True)

        model.vae.train()
        model.unet.train()

    def start_eval(
            self,
            model: StableDiffusionModel,
    ):
        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)
        if model.depth_estimator is not None:
            model.depth_estimator.to(self.train_device)

        model.vae.eval()
        model.unet.eval()
