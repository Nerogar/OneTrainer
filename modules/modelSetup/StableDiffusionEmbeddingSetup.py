from typing import Iterable

import torch
from diffusers.utils.import_utils import is_xformers_available
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionEmbeddingSetup(BaseModelSetup):
    all_token_embeds: Tensor
    all_original_token_embeds: Tensor
    trainable_token_embeds_mask: list[bool]
    untrainable_token_embeds_mask: list[bool]

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionEmbeddingSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        return model.text_encoder.get_input_embeddings().parameters()

    def setup_model(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
    ):
        model.text_encoder.requires_grad_(False)
        model.text_encoder.get_input_embeddings().requires_grad_(True)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        token_count = args.token_count if len(model.embeddings) == 0 else model.embeddings[0].token_count

        tokens = [f"<embedding_{i}>" for i in range(token_count)]
        model.tokenizer.add_tokens(tokens)
        model.text_encoder.resize_token_embeddings(len(model.tokenizer))

        with torch.no_grad():
            token_ids = model.tokenizer.encode(
                tokens,
                add_special_tokens=False,
            )

            self.all_token_embeds = model.text_encoder.get_input_embeddings().weight.data
            self.all_original_token_embeds = self.all_token_embeds.clone()
            self.trainable_token_embeds_mask = [(i in token_ids) for i in range(len(self.all_original_token_embeds))]
            self.untrainable_token_embeds_mask = [(i not in token_ids) for i in
                                                  range(len(self.all_original_token_embeds))]

            if len(model.embeddings) > 0:
                # an embedding was loaded
                for i, token_id in enumerate(token_ids):
                    self.all_token_embeds[token_id] = model.embeddings[0].vector[i]
            else:
                # create a new embedding
                initial_token_ids = model.tokenizer.encode(
                    args.initial_embedding_text,
                    add_special_tokens=False,
                    max_length=token_count,
                )
                pad_token_id = model.tokenizer.encode(
                    '*',
                    add_special_tokens=False,
                    max_length=token_count,
                )[0]
                initial_token_ids += [pad_token_id] * (token_count - len(initial_token_ids))
                for token_id, initial_token_id in zip(token_ids, initial_token_ids):
                    self.all_token_embeds[token_id] = self.all_token_embeds[initial_token_id]

                model.embeddings = [
                    StableDiffusionModelEmbedding("*", self.all_token_embeds[self.trainable_token_embeds_mask],
                                                  token_count)]

        if model.optimizer_state_dict is not None and model.optimizer is None:
            model.optimizer = create.create_optimizer(self.create_parameters_for_optimizer(model, args), args)
            # TODO: this will break if the optimizer class changed during a restart
            model.optimizer.load_state_dict(model.optimizer_state_dict)
            del model.optimizer_state_dict
        elif model.optimizer_state_dict is None and model.optimizer is None:
            model.optimizer = create.create_optimizer(self.create_parameters_for_optimizer(model, args), args)

    def setup_eval_device(
            self,
            model: StableDiffusionModel
    ):
        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device)
        model.unet.to(self.train_device)
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
        model.text_encoder.to(self.train_device)
        model.vae.to(self.temp_device)
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

        model.vae.enable_gradient_checkpointing()
        model.unet.enable_gradient_checkpointing()

        model.text_encoder.train()
        model.vae.eval()
        model.unet.train()

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
                scaled_latent_image.shape, generator=generator, device=args.train_device, dtype=args.train_dtype.torch_dtype()
            )
            offset_noise = torch.randn(
                scaled_latent_image.shape[0], scaled_latent_image.shape[1], 1, 1,
                generator=generator, device=args.train_device, dtype=args.train_dtype.torch_dtype()
            )
            latent_noise = normal_noise + (args.offset_noise_weight * offset_noise)
        else:
            latent_noise = torch.randn(
                scaled_latent_image.shape, generator=generator, device=args.train_device, dtype=args.train_dtype.torch_dtype()
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

        text_encoder_output = model.text_encoder(batch['tokens'], return_dict=True)[0]

        if args.model_type.has_mask_input() and args.model_type.has_conditioning_image_input():
            latent_input = torch.concat(
                [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1)
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
                noise = model.vae.decode(latent_noise / model.vae.scaling_factor).sample
                noise = noise.clamp(-1, 1)
                self.save_image(noise, args.debug_dir + "/training_batches", "1-noise", train_progress.global_step)

                # predicted noise
                predicted_noise = model.vae.decode(predicted_latent_noise / model.vae.scaling_factor).sample
                predicted_noise = predicted_noise.clamp(-1, 1)
                self.save_image(predicted_noise, args.debug_dir + "/training_batches", "2-predicted_noise",
                                train_progress.global_step)

                # noisy image
                noisy_latent_image = scaled_noisy_latent_image / model.vae.scaling_factor
                noisy_image = model.vae.decode(noisy_latent_image).sample
                noisy_image = noisy_image.clamp(-1, 1)
                self.save_image(noisy_image, args.debug_dir + "/training_batches", "3-noisy_image",
                                train_progress.global_step)

                # predicted image
                alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(args.train_device)
                sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                scaled_predicted_latent_image = \
                    (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) / sqrt_alpha_prod
                predicted_latent_image = scaled_predicted_latent_image / model.vae.scaling_factor
                predicted_image = model.vae.decode(predicted_latent_image).sample
                predicted_image = predicted_image.clamp(-1, 1)
                self.save_image(
                    predicted_image, args.debug_dir + "/training_batches", "4-predicted_image",
                    model.train_progress.global_step
                )

                # image
                image = model.vae.decode(latent_image).sample
                image = image.clamp(-1, 1)
                self.save_image(
                    image, args.debug_dir + "/training_batches", "5-image",
                    model.train_progress.global_step
                )

                # conditioning image
                if args.model_type.has_conditioning_image_input():
                    conditioning_image = model.vae.decode(latent_conditioning_image).sample
                    conditioning_image = conditioning_image.clamp(-1, 1)
                    self.save_image(
                        conditioning_image, args.debug_dir + "/training_batches", "6-conditioning_image",
                        train_progress.global_step
                    )

        return predicted_latent_noise, latent_noise

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        # reset untrainable embeddings
        with torch.no_grad():
            model.text_encoder.get_input_embeddings().weight[
                self.untrainable_token_embeds_mask
            ] = self.all_original_token_embeds[self.untrainable_token_embeds_mask]

        # save back to model
        model.embeddings = [StableDiffusionModelEmbedding(
            "*", self.all_token_embeds[self.trainable_token_embeds_mask], model.embeddings[0].token_count
        )]
