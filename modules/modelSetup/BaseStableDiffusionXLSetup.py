from abc import ABCMeta
from random import Random

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
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
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseStableDiffusionXLSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        if args.attention_mechanism == AttentionMechanism.DEFAULT:
            model.unet.set_attn_processor(AttnProcessor())
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
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_transformer_blocks(model.unet)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2)

    def __encode_text(
            self,
            model: StableDiffusionXLModel,
            text_encoder_layer_skip: int,
            text_encoder_2_layer_skip: int,
            tokens_1: Tensor = None,
            tokens_2: Tensor = None,
            text_encoder_1_output: Tensor = None,
            text_encoder_2_output: Tensor = None,
            pooled_text_encoder_2_output: Tensor = None,
            text: str = None,
    ):
        if tokens_1 is None and text is not None:
            tokenizer_output = model.tokenizer_1(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_1 = tokenizer_output.input_ids.to(model.text_encoder_1.device)

        if tokens_2 is None and text is not None:
            tokenizer_output = model.tokenizer_2(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            tokens_2 = tokenizer_output.input_ids.to(model.text_encoder_2.device)

        if text_encoder_1_output is None:
            text_encoder_1_output = model.text_encoder_1(
                tokens_1, output_hidden_states=True, return_dict=True
            )
            text_encoder_1_output = text_encoder_1_output.hidden_states[-(2 + text_encoder_layer_skip)]

        if text_encoder_2_output is None or pooled_text_encoder_2_output is None:
            text_encoder_2_output = model.text_encoder_2(
                tokens_2, output_hidden_states=True, return_dict=True
            )
            pooled_text_encoder_2_output = text_encoder_2_output.text_embeds
            text_encoder_2_output = text_encoder_2_output.hidden_states[-(2 + text_encoder_2_layer_skip)]

        text_encoder_output = torch.concat([text_encoder_1_output, text_encoder_2_output], dim=-1)

        return text_encoder_output, pooled_text_encoder_2_output

    def predict(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)
        rand = Random(train_progress.global_step)

        is_align_prop_step = args.align_prop and (rand.random() < args.align_prop_probability)

        vae_scaling_factor = model.vae.config['scaling_factor']

        text_encoder_output, pooled_text_encoder_2_output = self.__encode_text(
            model,
            args.text_encoder_layer_skip,
            args.text_encoder_2_layer_skip,
            tokens_1=batch['tokens_1'],
            tokens_2=batch['tokens_2'],
            text_encoder_1_output=batch[
                'text_encoder_1_hidden_state'] if not args.train_text_encoder and args.training_method != TrainingMethod.EMBEDDING else None,
            text_encoder_2_output=batch[
                'text_encoder_2_hidden_state'] if not args.train_text_encoder_2 and args.training_method != TrainingMethod.EMBEDDING else None,
            pooled_text_encoder_2_output=batch[
                'text_encoder_2_pooled_state'] if not args.train_text_encoder_2 and args.training_method != TrainingMethod.EMBEDDING else None,
        )

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image * vae_scaling_factor

        scaled_latent_conditioning_image = None
        if args.model_type.has_conditioning_image_input():
            scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

        latent_noise = self._create_noise(scaled_latent_image, args, generator)

        if is_align_prop_step and not deterministic:
            dummy = torch.zeros((1,), device=self.train_device)
            dummy.requires_grad_(True)

            negative_text_encoder_output, negative_pooled_text_encoder_2_output = self.__encode_text(
                model,
                args.text_encoder_layer_skip,
                args.text_encoder_2_layer_skip,
                text="",
            )
            negative_text_encoder_output = negative_text_encoder_output \
                .expand((scaled_latent_image.shape[0], -1, -1))
            negative_pooled_text_encoder_2_output = negative_pooled_text_encoder_2_output \
                .expand((scaled_latent_image.shape[0], -1))

            model.noise_scheduler.set_timesteps(args.align_prop_steps)

            scaled_noisy_latent_image = latent_noise

            timestep_high = int(args.align_prop_steps * args.max_noising_strength)
            timestep_low = \
                int(args.align_prop_steps * args.max_noising_strength * (1.0 - args.align_prop_truncate_steps))

            truncate_timestep_index = args.align_prop_steps - rand.randint(timestep_low, timestep_high)

            # original size of the image
            original_height = scaled_noisy_latent_image.shape[2] * 8
            original_width = scaled_noisy_latent_image.shape[3] * 8
            crops_coords_top = 0
            crops_coords_left = 0
            target_height = scaled_noisy_latent_image.shape[2] * 8
            target_width = scaled_noisy_latent_image.shape[3] * 8

            add_time_ids = torch.tensor([
                original_height,
                original_width,
                crops_coords_top,
                crops_coords_left,
                target_height,
                target_width
            ]).unsqueeze(0).expand((scaled_latent_image.shape[0], -1))

            add_time_ids = add_time_ids.to(
                dtype=scaled_noisy_latent_image.dtype,
                device=scaled_noisy_latent_image.device,
            )

            added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
            negative_added_cond_kwargs = {"text_embeds": negative_pooled_text_encoder_2_output,
                                          "time_ids": add_time_ids}

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

                predicted_latent_noise = checkpointed_unet(
                    sample=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=text_encoder_output,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                negative_predicted_latent_noise = checkpointed_unet(
                    sample=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_text_encoder_output,
                    added_cond_kwargs=negative_added_cond_kwargs,
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
                        predicted_image = self._project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1)
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
            if not deterministic:
                timestep = torch.randint(
                    low=0,
                    high=int(model.noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
                    size=(scaled_latent_image.shape[0],),
                    generator=generator,
                    device=scaled_latent_image.device,
                ).long()
            else:
                # -1 is for zero-based indexing
                timestep = torch.tensor(
                    int(model.noise_scheduler.config['num_train_timesteps'] * 0.5) - 1,
                    dtype=torch.long,
                    device=scaled_latent_image.device,
                )

            scaled_noisy_latent_image = model.noise_scheduler.add_noise(
                original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
            )

            # original size of the image
            original_height = batch['original_resolution'][0]
            original_width = batch['original_resolution'][1]
            crops_coords_top = batch['crop_offset'][0]
            crops_coords_left = batch['crop_offset'][1]
            target_height = batch['crop_resolution'][0]
            target_width = batch['crop_resolution'][1]

            add_time_ids = torch.stack([
                original_height,
                original_width,
                crops_coords_top,
                crops_coords_left,
                target_height,
                target_width
            ], dim=1)

            add_time_ids = add_time_ids.to(
                dtype=scaled_noisy_latent_image.dtype,
                device=scaled_noisy_latent_image.device,
            )

            if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}
            predicted_latent_noise = model.unet(
                sample=latent_input,
                timestep=timestep,
                encoder_hidden_states=text_encoder_output,
                added_cond_kwargs=added_cond_kwargs,
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

        if args.debug_mode:
            with torch.no_grad():
                # noise
                self._save_image(
                    self._project_latent_to_image(latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "1-noise",
                    train_progress.global_step
                )

                # predicted noise
                self._save_image(
                    self._project_latent_to_image(predicted_latent_noise).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "2-predicted_noise",
                    train_progress.global_step
                )

                # noisy image
                self._save_image(
                    self._project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1),
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
                    (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                    / sqrt_alpha_prod
                self._save_image(
                    self._project_latent_to_image(scaled_predicted_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "4-predicted_image",
                    model.train_progress.global_step
                )

                # image
                self._save_image(
                    self._project_latent_to_image(scaled_latent_image).clamp(-1, 1),
                    args.debug_dir + "/training_batches",
                    "5-image",
                    model.train_progress.global_step
                )

        return model_output_data

    def calculate_loss(
            self,
            model: StableDiffusionXLModel,
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
