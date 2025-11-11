from modules.util.config.TrainConfig import TrainConfig

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

OVERRIDE_CONFIG_KEYS = (
    # noise
    "offset_noise_weight",
    "perturbation_noise_weight",

    # timestep distribution
    "min_noising_strength",
    "max_noising_strength",
    "noising_weight",
    "noising_bias",
    "timestep_shift",
)


class PrepareBatchConfig(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, config: TrainConfig, config_overrides_in_name: str, config_out_name: str):
        super().__init__()
        self.config = config
        self.config_overrides_in_name = config_overrides_in_name
        self.config_out_name = config_out_name

    def length(self) -> int:
        return self._get_previous_length(self.config_overrides_in_name)

    def get_inputs(self) -> list[str]:
        return [self.config_overrides_in_name]

    def get_outputs(self) -> list[str]:
        return [self.config_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        config = {k: getattr(self.config, k) for k in OVERRIDE_CONFIG_KEYS}

        overrides: dict = self._get_previous_item(variation, self.config_overrides_in_name, index)
        config.update(overrides)

        return {
            self.config_out_name: config
        }


class ConceptConfigOverride(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, concept_index_in_name: str, concepts_meta_in_name: str, config_overrides_out_name: str):
        super().__init__()
        self.concept_index_in_name = concept_index_in_name
        self.concepts_meta_in_name = concepts_meta_in_name

        self.config_overrides_out_name = config_overrides_out_name

        self.concept_overrides = []

    def length(self) -> int:
        return self._get_previous_length(self.concept_index_in_name)

    def get_inputs(self) -> list[str]:
        return [self.concept_index_in_name]

    def get_outputs(self) -> list[str]:
        return [self.config_overrides_out_name]

    def start(self, variation: int):
        concept_dicts: list[dict] = self._get_previous_meta(variation, self.concepts_meta_in_name)

        for concept in concept_dicts:
            concept_noise: dict = concept["noise"]

            overrides = {
                k: v for k in OVERRIDE_CONFIG_KEYS
                if (v := concept_noise.get(k)) is not None
            }

            self.concept_overrides.append(overrides)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        concept_index: int = self._get_previous_item(variation, self.concept_index_in_name, index)
        return {
            self.config_overrides_out_name: self.concept_overrides[concept_index]
        }


class FileConfigOverride(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, file_config_in_name: str, config_overrides_name: str):
        super().__init__()
        self.file_config_in_name = file_config_in_name
        self.config_overrides_name = config_overrides_name

        self.__types: dict[str, type] = TrainConfig.default_values().types

    def length(self) -> int:
        return self._get_previous_length(self.file_config_in_name)

    def get_inputs(self) -> list[str]:
        return [self.file_config_in_name, self.config_overrides_name]

    def get_outputs(self) -> list[str]:
        return [self.config_overrides_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        overrides: dict = self._get_previous_item(variation, self.config_overrides_name, index)

        file_config = self._get_previous_item(variation, self.file_config_in_name, index)
        if isinstance(file_config, dict) and file_config:
            file_overrides = []
            for k in OVERRIDE_CONFIG_KEYS:
                v = file_config.get(k)
                if v is not None:
                    try:
                        # Cast value to expected type.
                        # Round floats to avoid overwhelming the timestep cache with small differences.
                        t = self.__types[k]
                        v = round(t(v), 6) if t is float else t(v)
                        file_overrides.append((k, v))
                    except Exception as ex:
                        print(f"File config override failed for key '{k}': {ex}")

            if file_overrides:
                overrides = overrides.copy()
                overrides.update(file_overrides)

        # debug randomization
        # else:
        #     import random
        #     overrides = overrides.copy()
        #     overrides.update( (k, random.random()) for k in CONFIG_KEYS_TIMESTEP if k != "max_noising_strength")

        return {
            self.config_overrides_name: overrides
        }
