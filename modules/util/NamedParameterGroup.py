from collections.abc import Iterable
from functools import cached_property

from modules.util.config.TrainConfig import TrainConfig

from torch.nn import Parameter


class NamedParameterGroup:
    def __init__(
            self,
            unique_name: str,
            parameters: Iterable[Parameter],
            learning_rate: float,
            display_name: str | None = None,
    ):
        self.unique_name = unique_name
        self.display_name = display_name if display_name is not None else unique_name
        self.parameters = list(parameters)
        self.learning_rate = learning_rate


class NamedParameterGroupCollection:
    __groups: list[NamedParameterGroup]

    def __init__(self):
        self.__groups = []

    def add_group(self, group: NamedParameterGroup):
        self.__groups.append(group)

    def parameters(self) -> list[Parameter]:
        return [p for group in self.__groups for p in group.parameters]

    def parameters_for_optimizer(self, config: TrainConfig) -> list[dict]:
        parameters = []

        for group in self.__groups:
            # Determine the learning rate
            lr = group.learning_rate if group.learning_rate is not None else config.learning_rate
            lr = lr * ((config.learning_rate_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)) ** 0.5)

            # Create a parameter group for the text encoder
            parameters.append({
                'name': group.display_name,
                'params': list(group.parameters),
                'lr': lr,
                'initial_lr': lr,
            })

        return parameters

    @cached_property
    def unique_name_mapping(self) -> list[str]:
        return [group.unique_name for group in self.__groups]

    @cached_property
    def display_name_mapping(self) -> list[str]:
        return [group.display_name for group in self.__groups]
