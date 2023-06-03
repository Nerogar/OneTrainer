from abc import ABCMeta

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism


class BaseStableDiffusionSetup(BaseModelSetup, metaclass=ABCMeta):

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

        model.unet.enable_gradient_checkpointing()
        model.text_encoder.gradient_checkpointing_enable()

    def predict(
            self,
            model: StableDiffusionModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress
    ) -> (Tensor, Tensor):
        latent_image = batch['latent_image']
        scaled_latent_image = latent_image * model.vae.config['scaling_factor']

        latent_conditioning_image = None
        scaled_latent_conditioning_image = None
        if args.model_type.has_conditioning_image_input():
            latent_conditioning_image = batch['latent_conditioning_image']
            scaled_latent_conditioning_image = latent_conditioning_image * model.vae.config['scaling_factor']

        generator = torch.Generator(device=args.train_device)
        generator.manual_seed(train_progress.global_step)

        if args.offset_noise_weight > 0:
            normal_noise = torch.randn(
                scaled_latent_image.shape, generator=generator, device=args.train_device,
                dtype=args.train_dtype.torch_dtype()
            )
            offset_noise = torch.randn(
                scaled_latent_image.shape[0], scaled_latent_image.shape[1], 1, 1,
                generator=generator, device=args.train_device, dtype=args.train_dtype.torch_dtype()
            )
            latent_noise = normal_noise + (args.offset_noise_weight * offset_noise)
        else:
            latent_noise = torch.randn(
                scaled_latent_image.shape, generator=generator, device=args.train_device,
                dtype=args.train_dtype.torch_dtype()
            )

        timestep = torch.randint(
            low=0,
            high=int(model.noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
            size=(scaled_latent_image.shape[0],),
            device=scaled_latent_image.device,
        ).long()

        scaled_noisy_latent_image = model.noise_scheduler.add_noise(
            original_samples=scaled_latent_image, noise=latent_noise, timesteps=timestep
        )

        if args.text_encoder_layer_skip > 0:
            text_encoder_output = model.text_encoder(batch['tokens'], return_dict=True, output_hidden_states=True)
            final_layer_norm = model.text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
            )
        else:
            text_encoder_output = model.text_encoder(batch['tokens'], return_dict=True)
            text_encoder_output = text_encoder_output.last_hidden_state

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
            predicted_latent_noise = model.unet(latent_input, timestep, text_encoder_output).sample

        if args.debug_mode:
            with torch.no_grad():
                # noise
                noise = model.vae.decode(latent_noise / model.vae.config['scaling_factor']).sample
                noise = noise.clamp(-1, 1)
                self.save_image(
                    noise,
                    args.debug_dir + "/training_batches",
                    "1-noise",
                    train_progress.global_step
                )

                # predicted noise
                predicted_noise = model.vae.decode(predicted_latent_noise / model.vae.config['scaling_factor']).sample
                predicted_noise = predicted_noise.clamp(-1, 1)
                self.save_image(
                    predicted_noise,
                    args.debug_dir + "/training_batches",
                    "2-predicted_noise",
                    train_progress.global_step
                )

                # noisy image
                noisy_latent_image = scaled_noisy_latent_image / model.vae.config['scaling_factor']
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
                    (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                predicted_latent_image = scaled_predicted_latent_image / model.vae.config['scaling_factor']
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

        if model.noise_scheduler.config.prediction_type == 'epsilon':
            return predicted_latent_noise, latent_noise
        elif model.noise_scheduler.config.prediction_type == 'v_prediction':
            predicted_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
            return predicted_latent_noise, predicted_velocity

