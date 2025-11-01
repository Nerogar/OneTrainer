from modules.util.config.ConfigOverride import ConfigOverrideSection, apply_overrides

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ConceptConfigOverride(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            concept_index_in_name: str,
            concepts_meta_in_name: str,
            config_overrides_out_name: str,
    ):
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
            overrides = {}
            concept_noise: dict = concept["noise"]

            if concept_noise["enable_noise_override"]:
                apply_overrides(overrides, concept_noise, ConfigOverrideSection.noise)

            if concept_noise["enable_timestep_distribution_override"]:
                apply_overrides(overrides, concept_noise, ConfigOverrideSection.timestep_distribution)

            self.concept_overrides.append(overrides)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        concept_index: int = self._get_previous_item(variation, self.concept_index_in_name, index)
        return {
            self.config_overrides_out_name: self.concept_overrides[concept_index]
        }
