from enum import Enum
from typing import Any, Sequence, Generic, TypeVar, cast

from modules.util.enum.TimestepDistribution import TimestepDistribution


class ConfigOverrideSection(Enum):
    """
    The string values of this enum are used as keys in the config_overrides dict in the pipeline.
    The enum names are lower-case so they can be used as in_names for PipelineModules, e.g. 'config_overrides.noise'.
    """

    noise = "noise"
    timestep_distribution = "timestep_distribution"

    def __str__(self):
        return self.value


# Must match name and type in TrainConfig
CONFIG_OVERRIDE_SECTION_KEYS: dict[ConfigOverrideSection, Sequence[tuple[str, type]]] = {
    ConfigOverrideSection.noise: (
        ("offset_noise_weight", float),
        ("generalized_offset_noise", bool),
        ("perturbation_noise_weight", float),
    ),
    ConfigOverrideSection.timestep_distribution: (
        ("timestep_distribution", TimestepDistribution),
        ("min_noising_strength", float),
        ("max_noising_strength", float),
        ("noising_weight", float),
        ("noising_bias", float),
        ("timestep_shift", float),
        ("dynamic_timestep_shifting", bool),
    ),
}


def apply_overrides(target: dict, source: dict, section: ConfigOverrideSection):
    "The values from 'source' are cast to the actual types in the config."

    new_overrides = []
    for k, t in CONFIG_OVERRIDE_SECTION_KEYS[section]:
        val = source.get(k)
        if val is not None:
            try:
                new_overrides.append((k, t(val)))
            except Exception as ex:
                print(f"Could not override '{k}' setting: {ex}")

    if new_overrides:
        old_overrides = target.get(str(section))
        if old_overrides is None:
            target[str(section)] = old_overrides = {}
        old_overrides.update(new_overrides)


T = TypeVar("T")
U = TypeVar("U")

class ConfigOverride(Generic[T]):
    def __init__(self, config: T, overrides: dict):
        super().__setattr__("__frozen", False)
        self.__config = config
        self.__overrides = overrides
        super().__setattr__("__frozen", True)

    def __getattr__(self, name: str) -> Any:
        val = self.__overrides.get(name)
        return getattr(self.__config, name) if val is None else val

    def __setattr__(self, name: str, value: Any):
        if getattr(self, "__frozen"):
            raise TypeError("Cannot set values on a wrapped config with overrides.")
        super().__setattr__(name, value)

    def as_original_type(self) -> T:
        return cast(T, self)

    @staticmethod
    def batch_apply(
            config: U,
            config_overrides: list[dict] | None,
            override_section: ConfigOverrideSection
    ) -> list[U] | None:
        "Returns a list of wrapped config files, or None if no overrides exist."

        if config_overrides is None:
            return None

        configs: list[U] = []
        has_override = False

        for sample_overrides in config_overrides:
            overrides = sample_overrides.get(str(override_section))
            if overrides:
                configs.append(ConfigOverride(config, overrides).as_original_type())
                has_override = True
            else:
                configs.append(config)

        return configs if has_override else None

    @classmethod
    def batch_debug_print(cls, config, config_overrides: list[dict], titles: list[str], section: ConfigOverrideSection):
        configs = cls.batch_apply(config, config_overrides, section)
        if configs is None:
            return

        print()
        print(f"{section} overrides:")
        for i, (cfg, overrides, title) in enumerate(zip(configs, config_overrides, titles)):
            if isinstance(cfg, ConfigOverride):
                print(f"[{i}] {title}:")
                for k in overrides[str(section)].keys():
                    v = getattr(cfg, k)
                    print(f"    {k} = {v}")
            else:
                print(f"[{i}] {title}: Global Settings")
