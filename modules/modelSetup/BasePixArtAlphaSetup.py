from abc import ABCMeta
from random import Random

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0, Attention
from diffusers.utils import is_xformers_available
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionNoiseMixin import ModelSetupDiffusionNoiseMixin
from modules.modelSetup.stableDiffusion.checkpointing_util import create_checkpointed_forward
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod


class BasePixArtAlphaSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    metaclass=ABCMeta,
):

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super(BasePixArtAlphaSetup, self).__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
    ):
        if args.attention_mechanism == AttentionMechanism.DEFAULT:
            for name, child_module in model.transformer.named_modules():
                if isinstance(child_module, Attention):
                    child_module.set_processor(AttnProcessor())
        elif args.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                for name, child_module in model.transformer.named_modules():
                    if isinstance(child_module, Attention):
                        child_module.set_processor(XFormersAttnProcessor())
                model.vae.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention_mechanism == AttentionMechanism.SDP:
            for name, child_module in model.transformer.named_modules():
                if isinstance(child_module, Attention):
                    child_module.set_processor(AttnProcessor2_0())

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
            model.transformer.enable_gradient_checkpointing()
            if args.train_text_encoder:
                model.text_encoder.encoder.gradient_checkpointing = True

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, args.train_dtype, [
            args.prior_weight_dtype,
            args.text_encoder_weight_dtype,
            args.vae_weight_dtype,
            args.lora_weight_dtype if args.training_method == TrainingMethod.LORA else None,
            args.embedding_weight_dtype if args.training_method == TrainingMethod.EMBEDDING else None,
        ])

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            args.train_dtype,
            args.fallback_train_dtype,
            [
                args.text_encoder_weight_dtype,
                args.lora_weight_dtype if args.training_method == TrainingMethod.LORA else None,
                args.embedding_weight_dtype if args.training_method == TrainingMethod.EMBEDDING else None,
            ],
        )


    def __encode_text(
            self,
            model: PixArtAlphaModel,
            args: TrainArgs,
            tokens: Tensor = None,
            attention_mask: Tensor = None,
            text: str = None,
    ):
        if tokens is None:
            tokenizer_output = model.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=120,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(model.text_encoder.device)

            attention_mask = tokenizer_output.attention_mask
            attention_mask = attention_mask.to(model.text_encoder.device)

        with model.text_encoder_autocast_context:
            text_encoder_output = model.text_encoder(
                tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            text_encoder_output.hidden_states = text_encoder_output.hidden_states[:-1]
            final_layer_norm = model.text_encoder.encoder.final_layer_norm
            prompt_embeds = final_layer_norm(
                text_encoder_output.hidden_states[-(1 + args.text_encoder_layer_skip)]
            )

        return prompt_embeds, attention_mask

    def predict(
            self,
            model: PixArtAlphaModel,
            batch: dict,
            args: TrainArgs,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            generator = torch.Generator(device=args.train_device)
            generator.manual_seed(train_progress.global_step)
            rand = Random(train_progress.global_step)

            is_align_prop_step = args.align_prop and (rand.random() < args.align_prop_probability)

            vae_scaling_factor = model.vae.config['scaling_factor']

            if args.train_text_encoder or args.training_method == TrainingMethod.EMBEDDING:
                text_encoder_output, text_encoder_attention_mask = self.__encode_text(
                    model,
                    args,
                    tokens=batch['tokens'],
                    attention_mask=batch['tokens_mask'],
                )
            else:
                text_encoder_output = batch['text_encoder_hidden_state']
                text_encoder_attention_mask = batch['tokens_mask']

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if args.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

            latent_noise = self._create_noise(scaled_latent_image, args, generator)

            if is_align_prop_step and not deterministic:
                dummy = torch.zeros((1,), device=self.train_device)
                dummy.requires_grad_(True)

                negative_text_encoder_output, negative_text_encoder_attention_mask = self.__encode_text(
                    model,
                    args,
                    text="",
                )
                negative_text_encoder_output = negative_text_encoder_output\
                    .expand((scaled_latent_image.shape[0], -1, -1))
                negative_text_encoder_attention_mask = negative_text_encoder_attention_mask \
                    .expand((scaled_latent_image.shape[0], -1, -1))

                model.noise_scheduler.set_timesteps(args.align_prop_steps)

                scaled_noisy_latent_image = latent_noise

                timestep_high = int(args.align_prop_steps * args.max_noising_strength)
                timestep_low = \
                    int(args.align_prop_steps * args.max_noising_strength * (1.0 - args.align_prop_truncate_steps))

                batch_size = scaled_latent_image.shape[0]
                height = scaled_latent_image.shape[2] * 8
                width = scaled_latent_image.shape[3] * 8
                resolution = torch.tensor([height, width]).repeat(batch_size, 1)
                aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
                resolution = resolution.to(dtype=args.train_dtype.torch_dtype(), device=self.train_device)
                aspect_ratio = aspect_ratio.to(dtype=args.train_dtype.torch_dtype(), device=self.train_device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                truncate_timestep_index = args.align_prop_steps - rand.randint(timestep_low, timestep_high)

                checkpointed_transformer = create_checkpointed_forward(model.transformer, self.train_device)

                for step in range(args.align_prop_steps):
                    timestep = model.noise_scheduler.timesteps[step] \
                        .expand((scaled_latent_image.shape[0],)) \
                        .to(device=model.transformer.device)

                    if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                        latent_input = torch.concat(
                            [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                        )
                    else:
                        latent_input = scaled_noisy_latent_image

                    predicted_latent_noise = checkpointed_transformer(
                        latent_input.to(dtype=args.train_dtype.torch_dtype()),
                        encoder_hidden_states=text_encoder_output.to(dtype=args.train_dtype.torch_dtype()),
                        encoder_attention_mask=text_encoder_attention_mask.to(dtype=args.train_dtype.torch_dtype()),
                        timestep=timestep,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    negative_predicted_latent_noise = checkpointed_transformer(
                        latent_input.to(dtype=args.train_dtype.torch_dtype()),
                        encoder_hidden_states=negative_text_encoder_output.to(dtype=args.train_dtype.torch_dtype()),
                        encoder_attention_mask=negative_text_encoder_attention_mask.to(dtype=args.train_dtype.torch_dtype()),
                        timestep=timestep,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    cfg_grad = (predicted_latent_noise - negative_predicted_latent_noise)
                    cfg_predicted_latent_noise = negative_predicted_latent_noise + args.align_prop_cfg_scale * cfg_grad
                    cfg_predicted_latent_noise = cfg_predicted_latent_noise.chunk(2, dim=1)[0]

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
                    predicted_image.append(checkpoint(
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
                timestep = self._get_timestep_discrete(
                    model.noise_scheduler,
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    args,
                )

                scaled_noisy_latent_image = self._add_noise_discrete(
                    scaled_latent_image,
                    latent_noise,
                    timestep,
                    model.noise_scheduler.betas,
                )

                if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
                    latent_input = torch.concat(
                        [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                    )
                else:
                    latent_input = scaled_noisy_latent_image

                batch_size = latent_input.shape[0]
                height = latent_input.shape[2] * 8
                width = latent_input.shape[3] * 8
                resolution = torch.tensor([height, width]).repeat(batch_size, 1)
                aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
                resolution = resolution.to(dtype=args.train_dtype.torch_dtype(), device=self.train_device)
                aspect_ratio = aspect_ratio.to(dtype=args.train_dtype.torch_dtype(), device=self.train_device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                text_encoder_attention_mask = text_encoder_attention_mask.view(batch_size, -1)

                predicted_latent_noise, predicted_latent_var_values = model.transformer(
                    latent_input.to(dtype=args.train_dtype.torch_dtype()),
                    encoder_hidden_states=text_encoder_output.to(dtype=args.train_dtype.torch_dtype()),
                    encoder_attention_mask=text_encoder_attention_mask.to(dtype=args.train_dtype.torch_dtype()),
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample.chunk(2, dim=1)

                model_output_data = {
                    'loss_type': 'target',
                    'predicted': predicted_latent_noise,
                    'target': latent_noise,
                    'noisy_latent_image': scaled_noisy_latent_image,
                    'predicted_var_values': predicted_latent_var_values,
                    'timestep': timestep,
                    'scaled_latent_image': scaled_latent_image,
                }

            if self.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.tokenizer),
                        args.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

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
                            (scaled_noisy_latent_image - predicted_latent_noise
                             * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
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
                            conditioning_image = model.vae.decode(
                                scaled_latent_conditioning_image / vae_scaling_factor).sample
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
            model: PixArtAlphaModel,
            batch: dict,
            data: dict,
            args: TrainArgs,
    ) -> Tensor:
        return self._diffusion_losses(
            batch=batch,
            data=data,
            args=args,
            train_device=self.train_device,
            betas=model.noise_scheduler.betas.to(device=self.train_device),
        ).mean()
