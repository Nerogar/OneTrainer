from abc import ABCMeta

import torch
from torch import Tensor


class ModelSetupFlowMatchingMixin(metaclass=ABCMeta):

    def __init__(self):
        super(ModelSetupFlowMatchingMixin, self).__init__()
        self.__sigma = None
        self.__one_minus_sigma = None

    def _add_model_precondition(
            self,
            model_output: Tensor,
            model_input: Tensor,
            timestep_index: Tensor,
    ) -> Tensor:
        sigmas = self.__sigma[timestep_index].to(dtype=model_output.dtype)

        while sigmas.dim() < model_output.dim():
            sigmas = sigmas.unsqueeze(-1)

        return model_output * (-sigmas) + model_input

    # TODO: Not sure which version should be used.
    #       Sampling from FlowMatchEulerDiscreteScheduler timesteps or a uniform distribution
    # def _add_noise_discrete(
    #         self,
    #         scaled_latent_image: Tensor,
    #         latent_noise: Tensor,
    #         timestep_index: Tensor,
    #         sigmas: Tensor,
    #         timesteps: Tensor,
    # ) -> tuple[Tensor, Tensor]:
    #     if self.__sigma is None:
    #         self.__timesteps = timesteps.to(device=scaled_latent_image.device)
    #         self.__sigma = sigmas.to(device=scaled_latent_image.device, dtype=torch.float32)
    #         self.__one_minus_sigma = 1.0 - self.__sigma
    #
    #     orig_dtype = scaled_latent_image.dtype
    #
    #     sigmas = self.__sigma[timestep_index]
    #     one_minus_sigmas = self.__one_minus_sigma[timestep_index]
    #     timestep = self.__timesteps[timestep_index]
    #
    #     while sigmas.dim() < scaled_latent_image.dim():
    #         sigmas = sigmas.unsqueeze(-1)
    #         one_minus_sigmas = one_minus_sigmas.unsqueeze(-1)
    #
    #     scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
    #                                 + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas
    #
    #     return scaled_noisy_latent_image.to(dtype=orig_dtype), timestep

    def _add_noise_discrete(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep_index: Tensor,
            sigmas: Tensor,
            timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.__sigma is None:
            num_timesteps = timesteps.shape[-1]
            self.__timesteps = torch.arange(start=1, end=num_timesteps, step=1, dtype=torch.int32, device=scaled_latent_image.device)
            self.__sigma = self.__timesteps / num_timesteps
            self.__one_minus_sigma = 1.0 - self.__sigma

        orig_dtype = scaled_latent_image.dtype

        sigmas = self.__sigma[timestep_index]
        one_minus_sigmas = self.__one_minus_sigma[timestep_index]
        timestep = self.__timesteps[timestep_index]

        while sigmas.dim() < scaled_latent_image.dim():
            sigmas = sigmas.unsqueeze(-1)
            one_minus_sigmas = one_minus_sigmas.unsqueeze(-1)

        scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
                                    + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas

        return scaled_noisy_latent_image.to(dtype=orig_dtype), timestep
