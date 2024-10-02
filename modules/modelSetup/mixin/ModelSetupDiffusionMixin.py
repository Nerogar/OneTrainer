from abc import ABCMeta
from collections.abc import Callable

from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients

from torch import Tensor


class ModelSetupDiffusionMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.__coefficients = None

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
            alphas_cumprod_fun: Callable[[Tensor, int], Tensor],
    ) -> Tensor:
        orig_dtype = scaled_latent_image.dtype

        alphas_cumprod = alphas_cumprod_fun(timestep, scaled_latent_image.dim())

        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        scaled_noisy_latent_image = scaled_latent_image.to(dtype=sqrt_alphas_cumprod.dtype) * sqrt_alphas_cumprod \
                                    + latent_noise.to(dtype=sqrt_alphas_cumprod.dtype) * sqrt_one_minus_alphas_cumprod

        return scaled_noisy_latent_image.to(dtype=orig_dtype)
