from abc import ABCMeta

import torch
from torch import Tensor


class ModelSetupFlowMatchingMixin(metaclass=ABCMeta):

    def __init__(self):
        super(ModelSetupFlowMatchingMixin, self).__init__()
        self.__sigma = None
        self.__one_minus_sigma = None

    def _add_noise_discrete(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep: Tensor,
            sigmas: Tensor,
    ) -> Tensor:
        if self.__sigma is None:
            self.__sigma = sigmas.to(device=scaled_latent_image.device, dtype=torch.float32)
            self.__one_minus_sigma = 1.0 - self.__sigma

        orig_dtype = scaled_latent_image.dtype

        sigmas = self.__sigma[timestep]
        one_minus_sigmas = self.__one_minus_sigma[timestep]

        while sigmas.dim() < scaled_latent_image.dim():
            sigmas = sigmas.unsqueeze(-1)
            one_minus_sigmas = one_minus_sigmas.unsqueeze(-1)

        scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
                                    + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas

        return scaled_noisy_latent_image.to(dtype=orig_dtype)
