from abc import ABCMeta

import numpy as np
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from torch import Tensor, Generator

from modules.util.config.TrainConfig import TrainConfig


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super(ModelSetupNoiseMixin, self).__init__()

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
            noise_scheduler: DDIMScheduler | FlowMatchEulerDiscreteScheduler,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            global_step: int,
    ) -> Tensor:
        if not deterministic:
            min_timestep = int(noise_scheduler.config['num_train_timesteps'] * config.min_noising_strength)
            max_timestep = int(noise_scheduler.config['num_train_timesteps'] * config.max_noising_strength)
            if config.noising_weight == 0:
                return torch.randint(
                    low=min_timestep,
                    high=max_timestep,
                    size=(batch_size,),
                    generator=generator,
                    device=generator.device,
                ).long()
            else:
                rng = np.random.default_rng(global_step)
                weights = np.linspace(0, 1, max_timestep - min_timestep)
                weights = 1 / (1 + np.exp(-config.noising_weight * (weights - config.noising_bias))) # Sigmoid
                weights /= np.sum(weights)
                samples = rng.choice(np.arange(min_timestep, max_timestep), size=(batch_size,), p=weights)
                return torch.tensor(samples, dtype=torch.long, device=generator.device)
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
            config: TrainConfig,
            global_step: int,
    ) -> Tensor:
        if not deterministic:
            if config.noising_weight == 0:
                return (1 - torch.rand(
                    size=(batch_size,),
                    generator=generator,
                    device=generator.device,
                )) * (config.max_noising_strength - config.min_noising_strength) + config.min_noising_strength
            else:
                rng = np.random.default_rng(global_step)
                choices = np.linspace(np.finfo(float).eps, 1, 5000)  # Discretize range (0, 1]
                weights = 1 / (1 + np.exp(-config.noising_weight * (choices - config.noising_bias)))  # Sigmoid
                weights /= np.sum(weights)
                samples = rng.choice(choices, size=(batch_size,), p=weights)
                samples = samples * (config.max_noising_strength - config.min_noising_strength) + config.min_noising_strength
                return torch.tensor(samples, dtype=torch.float, device=generator.device)
        else:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )
