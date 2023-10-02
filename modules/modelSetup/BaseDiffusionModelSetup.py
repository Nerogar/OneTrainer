from abc import ABCMeta

import torch
from torch import Tensor, Generator

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.args.TrainArgs import TrainArgs


class BaseDiffusionModelSetup(BaseModelSetup, metaclass=ABCMeta):

    def create_noise(self, source_tensor: Tensor, args: TrainArgs, generator: Generator):
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
