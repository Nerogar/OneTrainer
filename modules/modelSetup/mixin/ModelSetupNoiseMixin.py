import math
from abc import ABCMeta

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super(ModelSetupNoiseMixin, self).__init__()

        self.__weights = None

    def _create_noise(
            self,
            source_tensor: Tensor,
            config: TrainConfig,
            generator: Generator
    ):
        noise = torch.randn(
            source_tensor.shape,
            generator=generator,
            device=config.train_device,
            dtype=source_tensor.dtype
        )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], 1, 1),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return noise

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        if deterministic:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)
        else:
            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)
            num_timestep = max_timestep - min_timestep

            if config.timestep_distribution == TimestepDistribution.UNIFORM:
                return torch.randint(
                    low=min_timestep,
                    high=max_timestep,
                    size=(batch_size,),
                    generator=generator,
                    device=generator.device,
                ).long()
            elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                if self.__weights is None:
                    bias = config.noising_bias + 0.5
                    weight = config.noising_weight

                    weights = torch.linspace(0, 1, num_timestep)
                    weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
                    self.__weights = weights

                samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True) + min_timestep
                return samples.to(dtype=torch.long, device=generator.device)
            # elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:  ## multinomial implementation
            #     if self.__weights is None:
            #         bias = config.noising_bias
            #         scale = config.noising_weight + 1.0
            #
            #         weights = torch.linspace(0, 1, num_timestep)
            #         weights = \
            #             (1.0 / (scale * math.sqrt(2.0 * torch.pi))) \
            #             * (1.0 / (weights * (1.0 - weights))) \
            #             * torch.exp(
            #                 -((torch.logit(weights) - bias) ** 2.0) / (2.0 * scale ** 2.0)
            #             )
            #         weights.nan_to_num_(0)
            #         self.__weights = weights
            #
            #     samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True) + min_timestep
            #     return samples.to(dtype=torch.long, device=generator.device)
            elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                bias = config.noising_bias
                scale = config.noising_weight + 1.0

                normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
                logit_normal = normal.sigmoid()
                return (logit_normal * num_timestep + min_timestep).int()
            elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
                scale = config.noising_weight

                u = torch.rand(
                    size=(batch_size,),
                    generator=generator,
                    device=generator.device,
                )
                u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                return (u * num_timestep + min_timestep).int()
            elif config.timestep_distribution == TimestepDistribution.COS_MAP:
                if self.__weights is None:
                    weights = torch.linspace(0, 1, num_timestep)
                    weights = 2.0 / (math.pi - 2.0 * math.pi * weights + 2.0 * math.pi * weights ** 2.0)
                    self.__weights = weights

                samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True) + min_timestep
                return samples.to(dtype=torch.long, device=generator.device)

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        if deterministic:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = self._get_timestep_discrete(
                num_train_timesteps=discrete_timesteps,
                deterministic=False,
                generator=generator,
                batch_size=batch_size,
                config=config,
            ) + 1

            continuous = (discrete.float() / discrete_timesteps)
            return continuous
