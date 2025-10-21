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
            json_in_name: str,
            json_key_in_name: str,
            enabled_in_name: str,
            config_overrides_name: str,
    ):
        super().__init__()
        self.json_in_name = json_in_name
        self.json_key_in_name = json_key_in_name
        self.enabled_in_name = enabled_in_name

        self.config_overrides_name = config_overrides_name

    def length(self) -> int:
        return self._get_previous_length(self.json_in_name)

    def get_inputs(self) -> list[str]:
        return [self.json_in_name, self.json_key_in_name, self.enabled_in_name, self.config_overrides_name]

    def get_outputs(self) -> list[str]:
        return [self.config_overrides_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        overrides: dict = self._get_previous_item(variation, self.config_overrides_name, index)

        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        if enabled:
            overrides = self.__apply_json_overrides(overrides, variation, index)

        return {
            self.config_overrides_name: overrides
        }

    def __apply_json_overrides(self, overrides: dict, variation: int, index: int):
        json_data = self._get_previous_item(variation, self.json_in_name, index)
        if isinstance(json_data, dict) and json_data:
            json_key: str = self._get_previous_item(variation, self.json_key_in_name, index)
            json_overrides = self.__get_json_dict(json_data, json_key)

            if json_overrides:
                overrides = copy.deepcopy(overrides)

                if self.__check_enabled(json_overrides, "enable_noise_override"):
                    apply_overrides(overrides, json_overrides, ConfigOverrideSection.noise)

                if self.__check_enabled(json_overrides, "enable_timestep_distribution_override"):
                    apply_overrides(overrides, json_overrides, ConfigOverrideSection.timestep_distribution)

        return overrides

    def __get_json_dict(self, json_data, json_key: str) -> dict | None:
        json_key = json_key.strip()
        if not json_key:
            return json_data

        json_path = (key.strip() for key in json_key.split("."))
        for key in json_path:
            json_data = json_data.get(key)
            if not isinstance(json_data, dict):
                return None

        return json_data

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
