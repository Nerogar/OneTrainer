from abc import ABCMeta

import torch
from torch import Tensor


class ModelSetupFlowMatchingMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.__sigma = None
        self.__one_minus_sigma = None

    def _add_noise_discrete(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep: Tensor,
            timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.__sigma is None:
            num_timesteps = timesteps.shape[-1]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=scaled_latent_image.device)
            self.__sigma = all_timesteps / num_timesteps
            self.__one_minus_sigma = 1.0 - self.__sigma

        orig_dtype = scaled_latent_image.dtype

        sigmas = self.__sigma[timestep]
        one_minus_sigmas = self.__one_minus_sigma[timestep]

        while sigmas.dim() < scaled_latent_image.dim():
            sigmas = sigmas.unsqueeze(-1)
            one_minus_sigmas = one_minus_sigmas.unsqueeze(-1)

        scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
                                    + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas

        return scaled_noisy_latent_image.to(dtype=orig_dtype), sigmas
