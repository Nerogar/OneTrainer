from abc import ABCMeta
from random import Random

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionNoiseMixin import ModelSetupDiffusionNoiseMixin
from modules.modelSetup.stableDiffusion.checkpointing_util import \
    enable_checkpointing_for_transformer_blocks, enable_checkpointing_for_clip_encoder_layers, \
    create_checkpointed_unet_forward
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism


class BaseStableDiffusionSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    metaclass=ABCMeta,
):

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super(BaseStableDiffusionSetup, self).__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        if args.attention_mechanism == AttentionMechanism.DEFAULT:
            model.unet.set_attn_processor(AttnProcessor())
            pass
        elif args.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                model.unet.set_attn_processor(XFormersAttnProcessor())
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention_mechanism == AttentionMechanism.SDP:
            model.unet.set_attn_processor(AttnProcessor2_0())

            if is_xformers_available():
                try:
                    model.vae.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )

        if args.gradient_checkpointing:
            model.vae.enable_gradient_checkpointing()
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_transformer_blocks(model.unet)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder)

    def __encode_text(
            self,
            model: StableDiffusionModel,
            text_encoder_layer_skip: int,
            tokens: Tensor = None,
            text: str = None,
    ):
        if tokens is None:
            tokenizer_output = model.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(model.text_encoder.device)

        # TODO: use attention mask if this is true:
        # hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        text_encoder_output = model.text_encoder(tokens, return_dict=True, output_hidden_states=True)
        final_layer_norm = model.text_encoder.text_model.final_layer_norm
        return final_layer_norm(
            text_encoder_output.hidden_states[-(1 + text_encoder_layer_skip)]
        )

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress
    ) -> dict:
        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)
        rand = Random(train_progress.global_step)

        is_align_prop_step = args.align_prop and (rand.random() < args.align_prop_probability)

        vae_scaling_factor = model.vae.config['scaling_factor']

        text_encoder_output = self.__encode_text(
            model,
            args.text_encoder_layer_skip,
            tokens=batch['tokens'],
        )

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image * vae_scaling_factor

        scaled_latent_conditioning_image = None
        if args.model_type.has_conditioning_image_input():
            scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

        latent_noise = self._create_noise(scaled_latent_image, args, generator)

        if is_align_prop_step:
            dummy = torch.zeros((1,), device=self.train_device)
            dummy.requires_grad_(True)

            negative_text_encoder_output = self.__encode_text(
                model,
                args.text_encoder_layer_skip,
                text="",
            ).expand((scaled_latent_image.shape[0], -1, -1))

            model.noise_scheduler.set_timesteps(args.align_prop_steps)

            scaled_noisy_latent_image = latent_noise

            timestep_high = int(args.align_prop_steps * args.max_noising_strength)
            timestep_low = \
                int(args.align_prop_steps * args.max_noising_strength * (1.0 - args.align_prop_truncate_steps))

            truncate_timestep_index = args.align_prop_steps - rand.randint(timestep_low, timestep_high)

            checkpointed_unet = create_checkpointed_unet_forward(model.unet)

            for step in range(args.align_prop_steps):
                timestep = model.noise_scheduler.timesteps[step] \
                    .expand((scaled_latent_image.shape[0],)) \
                    .to(device=model.unet.device)

                if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                    latent_input = torch.concat(
                        [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                    )
                else:
                    latent_input = scaled_noisy_latent_image

                if args.model_type.has_depth_input():
                    predicted_latent_noise = checkpointed_unet(
                        latent_input,
                        timestep,
                        text_encoder_output,
                        batch['latent_depth'],
                    ).sample

                    negative_predicted_latent_noise = checkpointed_unet(
                        latent_input,
                        timestep,
                        negative_text_encoder_output,
                        batch['latent_depth'],
                    ).sample
                else:
                    predicted_latent_noise = checkpointed_unet(
                        latent_input,
                        timestep,
                        text_encoder_output,
                    ).sample

                    negative_predicted_latent_noise = checkpointed_unet(
                        latent_input,
                        timestep,
                        negative_text_encoder_output,
                    ).sample

                cfg_grad = (predicted_latent_noise - negative_predicted_latent_noise)
                cfg_predicted_latent_noise = negative_predicted_latent_noise + args.align_prop_cfg_scale * cfg_grad

                scaled_noisy_latent_image = model.noise_scheduler \
                    .step(cfg_predicted_latent_noise, timestep[0].long(), scaled_noisy_latent_image) \
                    .prev_sample

                if step < truncate_timestep_index:
                    scaled_noisy_latent_image = scaled_noisy_latent_image.detach()

                if self.debug_mode:
                    with torch.no_grad():
                        # predicted image
                        predicted_image = model.vae.decode(
                            scaled_noisy_latent_image.to(dtype=model.vae.dtype) / vae_scaling_factor).sample
                        predicted_image = predicted_image.clamp(-1, 1)
                        self._save_image(
                            predicted_image,
                            args.debug_dir + "/training_batches",
                            "2-predicted_image_" + str(step),
                            train_progress.global_step
                        )

            predicted_latent_image = scaled_noisy_latent_image / vae_scaling_factor
            predicted_latent_image = predicted_latent_image.to(dtype=model.vae.dtype)

            predicted_image = []
            for x in predicted_latent_image.split(1):
                predicted_image.append(torch.utils.checkpoint.checkpoint(
                    model.vae.decode,
                    x,
                    use_reentrant=False
                ).sample)
            predicted_image = torch.cat(predicted_image)

            model_output_data = {
                'loss_type': 'align_prop',
                'predicted': predicted_image,
            }
        else:
            timestep = torch.randint(
                low=0,
                high=int(model.noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
                size=(scaled_latent_image.shape[0],),
                generator=generator,
                device=scaled_latent_image.device,
            ).long()

            scaled_noisy_latent_image = model.noise_scheduler.add_noise(
                original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
            )

            if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            if args.model_type.has_depth_input():
                predicted_latent_noise = model.unet(
                    latent_input, timestep, text_encoder_output, batch['latent_depth']
                ).sample
            else:
                predicted_latent_noise = model.unet(
                    latent_input, timestep, text_encoder_output
                ).sample

            model_output_data = {}

            if model.noise_scheduler.config.prediction_type == 'epsilon':
                model_output_data = {
                    'loss_type': 'target',
                    'predicted': predicted_latent_noise,
                    'target': latent_noise,
                }
            elif model.noise_scheduler.config.prediction_type == 'v_prediction':
                target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
                model_output_data = {
                    'loss_type': 'target',
                    'predicted': predicted_latent_noise,
                    'target': target_velocity,
                }

        if self.debug_mode:
            with torch.no_grad():
                if is_align_prop_step:
                    # noise
                    noise = model.vae.decode(latent_noise / vae_scaling_factor).sample
                    noise = noise.clamp(-1, 1)
                    self._save_image(
                        noise,
                        args.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # image
                    image = model.vae.decode(scaled_latent_image / vae_scaling_factor).sample
                    image = image.clamp(-1, 1)
                    self._save_image(
                        image,
                        args.debug_dir + "/training_batches",
                        "2-image",
                        train_progress.global_step
                    )
                else:
                    # noise
                    noise = model.vae.decode(latent_noise / vae_scaling_factor).sample
                    noise = noise.clamp(-1, 1)
                    self._save_image(
                        noise,
                        args.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # predicted noise
                    predicted_noise = model.vae.decode(predicted_latent_noise / vae_scaling_factor).sample
                    predicted_noise = predicted_noise.clamp(-1, 1)
                    self._save_image(
                        predicted_noise,
                        args.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step
                    )

                    # noisy image
                    noisy_latent_image = scaled_noisy_latent_image / vae_scaling_factor
                    noisy_image = model.vae.decode(noisy_latent_image).sample
                    noisy_image = noisy_image.clamp(-1, 1)
                    self._save_image(
                        noisy_image,
                        args.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step
                    )

                    # predicted image
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(args.train_device)
                    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (
                                scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                    predicted_latent_image = scaled_predicted_latent_image / vae_scaling_factor
                    predicted_image = model.vae.decode(predicted_latent_image).sample
                    predicted_image = predicted_image.clamp(-1, 1)
                    self._save_image(
                        predicted_image,
                        args.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step
                    )

                    # image
                    image = model.vae.decode(latent_image).sample
                    image = image.clamp(-1, 1)
                    self._save_image(
                        image,
                        args.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step
                    )

                    # conditioning image
                    if args.model_type.has_conditioning_image_input():
                        conditioning_image = model.vae.decode(scaled_latent_conditioning_image / vae_scaling_factor).sample
                        conditioning_image = conditioning_image.clamp(-1, 1)
                        self._save_image(
                            conditioning_image,
                            args.debug_dir + "/training_batches",
                            "6-conditioning_image",
                            train_progress.global_step
                        )

        return model_output_data

    def calculate_loss(
            self,
            model: StableDiffusionModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
    ) -> Tensor:
        return self._diffusion_loss(
            batch=batch,
            data=data,
            args=args,
            train_device=self.train_device,
        )
