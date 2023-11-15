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
                prior_attention_blocks = [x for x in model.prior_prior.blocks if isinstance(x, AttnBlock)]
                for prior_attention_block in prior_attention_blocks:
                    prior_attention_block.attention.set_processor(XFormersAttnProcessor())
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention_mechanism == AttentionMechanism.SDP:
            prior_attention_blocks = [x for x in model.prior_prior.blocks if isinstance(x, AttnBlock)]
            for prior_attention_block in prior_attention_blocks:
                prior_attention_block.attention.set_processor(AttnProcessor2_0())

        # if args.gradient_checkpointing:
        #     model.unet.enable_gradient_checkpointing()
        #     enable_checkpointing_for_transformer_blocks(model.unet)
        #     enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1)
        #     enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2)

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
            train_progress: TrainProgress
    ) -> dict:
        latent_mean = 42.0
        latent_std = 1.0

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image.add(latent_std).div(latent_mean)

        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)

        latent_noise = self._create_noise(scaled_latent_image, args, generator)

        timestep = (1 - torch.rand(
            size=(scaled_latent_image.shape[0],),
            generator=generator,
            device=scaled_latent_image.device,
        )).mul(1.08).add(0.001).clamp(0.001, 1.0)

        scaled_noisy_latent_image = self.__add_noise(
            original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
        )

        if args.train_text_encoder or args.training_method == TrainingMethod.EMBEDDING:
            text_encoder_output = model.prior_text_encoder(
                batch['tokens_1'], output_hidden_states=True, return_dict=True
            )
            text_encoder_output = text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
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

        if args.masked_training and not args.model_type.has_conditioning_image_input():
            losses = loss_util.masked_loss(
                F.mse_loss,
                predicted,
                target,
                batch['latent_mask'],
                args.unmasked_weight,
                args.normalize_masked_area_loss
            ).mean([1, 2, 3])
        else:
            losses = F.mse_loss(
                predicted,
                target,
                reduction='none'
            ).mean([1, 2, 3])

            if args.normalize_masked_area_loss:
                clamped_mask = torch.clamp(batch['latent_mask'], args.unmasked_weight, 1)
                losses = losses / clamped_mask.mean(dim=(1, 2, 3))


        k = 1.0
        gamma = 1.0
        alpha_cumprod = self.__alpha_cumprod(target, timestep)
        loss_weight = (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma

        return (losses * loss_weight).mean()
