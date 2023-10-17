from abc import ABCMeta
from random import Random

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseDiffusionModelSetup import BaseDiffusionModelSetup
from modules.modelSetup.stableDiffusion.checkpointing_util import \
    enable_checkpointing_for_transformer_blocks, enable_checkpointing_for_clip_encoder_layers
from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.util import loss_util
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism


class BaseStableDiffusionSetup(BaseDiffusionModelSetup, metaclass=ABCMeta):

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super(BaseStableDiffusionSetup, self).__init__(train_device, temp_device, debug_mode)
        self.aesthetic_score_model = None

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

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress
    ) -> dict:
        # alignProp args
        num_timesteps = 20
        truncate_timesteps = 0.3
        cfg_scale = 7.0

        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)
        rand = Random(train_progress.global_step)

        is_align_prop_step = args.align_prop and (rand.random() < args.align_prop_probability)

        # TODO: use attention mask if this is true:
        # hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        text_encoder_output = model.text_encoder(batch['tokens'], return_dict=True, output_hidden_states=True)
        final_layer_norm = model.text_encoder.text_model.final_layer_norm
        text_encoder_output = final_layer_norm(
            text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
        )

        vae_scaling_factor = model.vae.config['scaling_factor']

        latent_image = batch['latent_image']
        scaled_latent_image = latent_image * vae_scaling_factor

        latent_conditioning_image = None
        scaled_latent_conditioning_image = None
        if args.model_type.has_conditioning_image_input():
            latent_conditioning_image = batch['latent_conditioning_image']
            scaled_latent_conditioning_image = latent_conditioning_image * vae_scaling_factor

        latent_noise = self.create_noise(scaled_latent_image, args, generator)

        if is_align_prop_step:
            negative_tokenizer_output = model.tokenizer(
                "",
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            negative_tokens = negative_tokenizer_output.input_ids.to(model.text_encoder.device)

            negative_text_encoder_output = model.text_encoder(
                negative_tokens, return_dict=True, output_hidden_states=True
            )
            negative_text_encoder_output = final_layer_norm(
                negative_text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
            ).expand((scaled_latent_image.shape[0], -1, -1))

            model.noise_scheduler.set_timesteps(num_timesteps)

            timestep_high = int(num_timesteps * args.max_noising_strength)
            timestep_low = int(num_timesteps * args.max_noising_strength * (1.0 - truncate_timesteps))

            timestep_index = torch.randint(
                low=timestep_low,
                high=timestep_high,
                size=(1,),
                generator=generator,
                device=scaled_latent_image.device,
            ).long().cpu().item()

            timestep = model.noise_scheduler.timesteps[-timestep_index - 1].expand((scaled_latent_image.shape[0],))

            scaled_noisy_latent_image = model.noise_scheduler.add_noise(
                original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
            )

            for step in range(timestep_index, -1, -1):
                timestep = model.noise_scheduler.timesteps[-step - 1] \
                    .expand((scaled_latent_image.shape[0],)) \
                    .to(device=model.unet.device)

                if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                    latent_input = torch.concat(
                        [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                    )
                else:
                    latent_input = scaled_noisy_latent_image

                if args.model_type.has_depth_input():
                    predicted_latent_noise = torch.utils.checkpoint.checkpoint(
                        model.unet,
                        latent_input,
                        timestep,
                        text_encoder_output,
                        batch['latent_depth'],
                        use_reentrant=False
                    ).sample

                    negative_predicted_latent_noise = torch.utils.checkpoint.checkpoint(
                        model.unet,
                        latent_input,
                        timestep,
                        negative_text_encoder_output,
                        batch['latent_depth'],
                        use_reentrant=False
                    ).sample
                else:
                    predicted_latent_noise = torch.utils.checkpoint.checkpoint(
                        model.unet,
                        latent_input,
                        timestep,
                        text_encoder_output,
                        use_reentrant=False
                    ).sample

                    negative_predicted_latent_noise = torch.utils.checkpoint.checkpoint(
                        model.unet,
                        latent_input,
                        timestep,
                        negative_text_encoder_output,
                        use_reentrant=False
                    ).sample

                cfg_grad = (predicted_latent_noise - negative_predicted_latent_noise)
                cfg_predicted_latent_noise = negative_predicted_latent_noise + cfg_scale * cfg_grad

                scaled_noisy_latent_image = model.noise_scheduler \
                    .step(cfg_predicted_latent_noise, timestep[0].long(), scaled_noisy_latent_image) \
                    .prev_sample

                if self.debug_mode:
                    with torch.no_grad():
                        # predicted image
                        predicted_image = model.vae.decode(
                            scaled_noisy_latent_image.to(dtype=model.vae.dtype) / vae_scaling_factor).sample
                        predicted_image = predicted_image.clamp(-1, 1)
                        self.save_image(
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
                    self.save_image(
                        noise,
                        args.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # image
                    image = model.vae.decode(scaled_latent_image / vae_scaling_factor).sample
                    image = image.clamp(-1, 1)
                    self.save_image(
                        image,
                        args.debug_dir + "/training_batches",
                        "2-image",
                        train_progress.global_step
                    )
                else:
                    # noise
                    noise = model.vae.decode(latent_noise / vae_scaling_factor).sample
                    noise = noise.clamp(-1, 1)
                    self.save_image(
                        noise,
                        args.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # predicted noise
                    predicted_noise = model.vae.decode(predicted_latent_noise / vae_scaling_factor).sample
                    predicted_noise = predicted_noise.clamp(-1, 1)
                    self.save_image(
                        predicted_noise,
                        args.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step
                    )

                    # noisy image
                    noisy_latent_image = scaled_noisy_latent_image / vae_scaling_factor
                    noisy_image = model.vae.decode(noisy_latent_image).sample
                    noisy_image = noisy_image.clamp(-1, 1)
                    self.save_image(
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
                    self.save_image(
                        predicted_image,
                        args.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step
                    )

                    # image
                    image = model.vae.decode(latent_image).sample
                    image = image.clamp(-1, 1)
                    self.save_image(
                        image,
                        args.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step
                    )

                    # conditioning image
                    if args.model_type.has_conditioning_image_input():
                        conditioning_image = model.vae.decode(latent_conditioning_image).sample
                        conditioning_image = conditioning_image.clamp(-1, 1)
                        self.save_image(
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
        loss_type = data['loss_type']
        if loss_type == 'align_prop':
            predicted = data['predicted']

            if self.aesthetic_score_model is None:
                self.aesthetic_score_model = AestheticScoreModel()
                self.aesthetic_score_model.to(device=self.train_device)
                self.aesthetic_score_model.requires_grad_(False)

            losses = self.aesthetic_score_model(predicted) * args.align_prop_weight
        else:
            predicted = data['predicted']
            target = data['target']

            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
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

        return losses.mean()
