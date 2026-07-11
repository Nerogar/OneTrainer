
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Tensor


class TimestepGenerator(ModelSetupNoiseMixin):

    def __init__(
            self,
            timestep_distribution: TimestepDistribution,
            min_noising_strength: float,
            max_noising_strength: float,
            noising_weight: float,
            noising_bias: float,
            timestep_shift: float,
    ):
        super().__init__()

        self.timestep_distribution = timestep_distribution
        self.min_noising_strength = min_noising_strength
        self.max_noising_strength = max_noising_strength
        self.noising_weight = noising_weight
        self.noising_bias = noising_bias
        self.timestep_shift = timestep_shift

    def generate(self) -> Tensor:
        generator = torch.Generator()
        generator.seed()

        config = TrainConfig.default_values()
        config.timestep_distribution = self.timestep_distribution
        config.min_noising_strength = self.min_noising_strength
        config.max_noising_strength = self.max_noising_strength
        config.noising_weight = self.noising_weight
        config.noising_bias = self.noising_bias
        config.timestep_shift = self.timestep_shift


        return self._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=1000000,
            config=config,
        )


class TimestepDistributionWindowController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_distribution_options(self) -> list[str]:
        return [str(x) for x in list(TimestepDistribution)]

    def generate_preview_data(self) -> Tensor:
        generator = TimestepGenerator(
            timestep_distribution=self.train_config.timestep_distribution,
            min_noising_strength=self.train_config.min_noising_strength,
            max_noising_strength=self.train_config.max_noising_strength,
            noising_weight=self.train_config.noising_weight,
            noising_bias=self.train_config.noising_bias,
            timestep_shift=self.train_config.timestep_shift,
        )
        return generator.generate()
