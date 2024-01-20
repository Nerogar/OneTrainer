from abc import ABCMeta
from typing import Callable

import torch
from diffusers import DDIMScheduler
from torch import Tensor, Generator

from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.args.TrainArgs import TrainArgs


class ModelSetupDiffusionNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super(ModelSetupDiffusionNoiseMixin, self).__init__()
        self.__coefficients = None

    def _create_noise(
            self,
            source_tensor: Tensor,
            args: TrainArgs,
            generator: Generator
    ):
        noise = torch.randn(
            source_tensor.shape,
            generator=generator,
            device=args.train_device,
            dtype=source_tensor.dtype
        )

        if args.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], 1, 1),
                generator=generator,
                device=args.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (args.offset_noise_weight * offset_noise)

        if args.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=args.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (args.perturbation_noise_weight * perturbation_noise)

        return noise

    def _get_timestep_discrete(
            self,
            noise_scheduler: DDIMScheduler,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            args: TrainArgs,
    ) -> Tensor:
        if not deterministic:
            return torch.randint(
                low=0,
                high=int(noise_scheduler.config['num_train_timesteps'] * args.max_noising_strength),
                size=(batch_size,),
                generator=generator,
                device=generator.device,
            ).long()
        else:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(noise_scheduler.config['num_train_timesteps'] * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            args: TrainArgs,
    ) -> Tensor:
        if not deterministic:
            return (1 - torch.rand(
                size=(batch_size,),
                generator=generator,
                device=generator.device,
            )) * args.max_noising_strength
        else:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )


    def _add_noise_discrete(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep: Tensor,
            betas: Tensor,
    ) -> Tensor:
        if self.__coefficients is None:
            betas = betas.to(device=scaled_latent_image.device)
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        orig_dtype = scaled_latent_image.dtype

        sqrt_alphas_cumprod = self.__coefficients.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alphas_cumprod = self.__coefficients.sqrt_one_minus_alphas_cumprod[timestep]

        while sqrt_alphas_cumprod.dim() < scaled_latent_image.dim():
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        scaled_noisy_latent_image = scaled_latent_image.to(dtype=sqrt_alphas_cumprod.dtype) * sqrt_alphas_cumprod \
                                    + latent_noise.to(dtype=sqrt_alphas_cumprod.dtype) * sqrt_one_minus_alphas_cumprod

        return scaled_noisy_latent_image.to(dtype=orig_dtype)

    def _add_noise_continuous(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep: Tensor,
            alpha_cumprod_fun: Callable[[Tensor, int], Tensor],
    ) -> Tensor:
        alpha_cumprod = alpha_cumprod_fun(timestep, scaled_latent_image.dim())

        return alpha_cumprod.sqrt() * scaled_latent_image + (1 - alpha_cumprod).sqrt() * latent_noise

