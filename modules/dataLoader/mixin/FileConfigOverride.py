import copy

from modules.util.config.ConfigOverride import ConfigOverrideSection, apply_overrides

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class FileConfigOverride(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            file_config_in_name: str,
            config_overrides_name: str,
    ):
        super().__init__()
        self.file_config_in_name = file_config_in_name
        self.config_overrides_name = config_overrides_name

    def length(self) -> int:
        return self._get_previous_length(self.file_config_in_name)

    def get_inputs(self) -> list[str]:
        return [self.file_config_in_name, self.config_overrides_name]

    def get_outputs(self) -> list[str]:
        return [self.config_overrides_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        overrides: dict = self._get_previous_item(variation, self.config_overrides_name, index)

        file_overrides = self._get_previous_item(variation, self.file_config_in_name, index)
        if isinstance(file_overrides, dict) and file_overrides:
            overrides = copy.deepcopy(overrides)

            if self.__check_enabled(file_overrides, "enable_noise_override"):
                apply_overrides(overrides, file_overrides, ConfigOverrideSection.noise)

            if self.__check_enabled(file_overrides, "enable_timestep_distribution_override"):
                apply_overrides(overrides, file_overrides, ConfigOverrideSection.timestep_distribution)

        return {
            self.config_overrides_name: overrides
        }

    @staticmethod
    def __check_enabled(overrides: dict, key: str) -> bool:
        # When the enable toggle is missing, it defaults to True.
        # This allows to disable the override for certain files,
        # but doesn't require adding the toggle to all files.
        val = overrides.get(key, True)
        if isinstance(val, str):
            val = val.lower()
            if val == "0" or val == "false":
                return False

        return bool(val)
