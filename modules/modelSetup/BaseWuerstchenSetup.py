from abc import ABCMeta

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.pipelines.wuerstchen.modeling_wuerstchen_common import AttnBlock
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionNoiseMixin import ModelSetupDiffusionNoiseMixin
from modules.modelSetup.stableDiffusion.checkpointing_util import enable_checkpointing_for_transformer_blocks, \
    enable_checkpointing_for_clip_encoder_layers
from modules.util import loss_util
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseWuerstchenSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    metaclass=ABCMeta,
):

    def setup_optimizations(
            self,
            model: WuerstchenModel,
            args: TrainArgs,
    ):
        if args.attention_mechanism == AttentionMechanism.DEFAULT:
            model.prior_prior.set_attn_processor(AttnProcessor())
        elif args.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                model.prior_prior.set_attn_processor(XFormersAttnProcessor())
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention_mechanism == AttentionMechanism.SDP:
            model.prior_prior.set_attn_processor(AttnProcessor2_0())

        if args.gradient_checkpointing:
            model.prior_prior.enable_gradient_checkpointing()
            enable_checkpointing_for_clip_encoder_layers(model.prior_text_encoder)

    def __alpha_cumprod(
            self,
            original_samples,
            timesteps,
    ):
        # copied and modified from https://github.com/dome272/wuerstchen
        s = torch.tensor([0.008]).to(device=original_samples.device)
        init_alpha_cumprod = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        alpha_cumprod = torch.cos((timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2 / init_alpha_cumprod
        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        alpha_cumprod = alpha_cumprod.view(timesteps.size(0), *[1 for _ in original_samples.shape[1:]])
        return alpha_cumprod

    def __add_noise(
            self,
            original_samples,
            noise,
            timesteps,
    ):
        alpha_cumprod = self.__alpha_cumprod(original_samples, timesteps)
        return alpha_cumprod.sqrt() * original_samples + (1 - alpha_cumprod).sqrt() * noise

    def predict(
            self,
            model: WuerstchenModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
            *,
            deterministic: bool = True,
    ) -> dict:
        latent_mean = 42.0
        latent_std = 1.0

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image.add(latent_std).div(latent_mean)

        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)

        latent_noise = self._create_noise(scaled_latent_image, args, generator)

        if not deterministic:
            timestep = (
                1 - torch.rand(
                    size=(scaled_latent_image.shape[0],),
                    generator=generator,
                    device=scaled_latent_image.device,
                )
            ).mul(1.08).add(0.001).clamp(0.001, 1.0)
        else:
            timestep = torch.full(
                size=(scaled_latent_image.shape[0],),
                fill_value=0.5,
                device=scaled_latent_image.device,
            )

        scaled_noisy_latent_image = self.__add_noise(
            original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
        )

        if args.train_text_encoder or args.training_method == TrainingMethod.EMBEDDING:
            text_encoder_output = model.prior_text_encoder(
                batch['tokens'], output_hidden_states=True, return_dict=True
            )
            final_layer_norm = model.prior_text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
            )
        else:
            text_encoder_output = batch['text_encoder_hidden_state']

        latent_input = scaled_noisy_latent_image

        predicted_latent_noise = model.prior_prior(latent_input, timestep, text_encoder_output)

        model_output_data = {
            'predicted': predicted_latent_noise,
            'target': latent_noise,
            'timestep': timestep,
        }

        if args.debug_mode:
            with torch.no_grad():
                # noise
                self.save_image(
                    self.project_latent_to_image(latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "1-noise",
                    train_progress.global_step
                )

                # predicted noise
                self.save_image(
                    self.project_latent_to_image(predicted_latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "2-predicted_noise",
                    train_progress.global_step
                )

                # noisy image
                self.save_image(
                    self.project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "3-noisy_image",
                    train_progress.global_step
                )

                # predicted image
                alpha_cumprod = self.__alpha_cumprod(latent_noise, timestep)
                sqrt_alpha_prod = alpha_cumprod ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                scaled_predicted_latent_image = \
                    (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                    / sqrt_alpha_prod
                self.save_image(
                    self.project_latent_to_image(scaled_predicted_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "4-predicted_image",
                    model.train_progress.global_step
                )

                # image
                self.save_image(
                    self.project_latent_to_image(scaled_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "5-image",
                    model.train_progress.global_step
                )

        return model_output_data

    def calculate_loss(
            self,
            model: WuerstchenModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
    ) -> Tensor:
        predicted = data['predicted']
        target = data['target']
        timestep = data['timestep']

        losses = F.mse_loss(
            predicted,
            target,
            reduction='none'
        ).mean([1, 2, 3])

        k = 1.0
        gamma = 1.0
        alpha_cumprod = self.__alpha_cumprod(target, timestep)
        loss_weight = (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma

        return (losses * loss_weight).mean()
