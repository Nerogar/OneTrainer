from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.ConceptConfig import ConceptConfig

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class StoreConcepts(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, config: TrainConfig, concepts_meta_in_name: str):
        super(StoreConcepts, self).__init__()
        self.config = config
        self.concepts_meta_in_name = concepts_meta_in_name

    def length(self) -> int:
        return 0

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return []

    def start(self, variation: int):
        concept_dicts: list[dict] = self._get_previous_meta(variation, self.concepts_meta_in_name)

        concepts = [
            ConceptConfig.default_values().from_dict(concept_dict)
            for concept_dict in concept_dicts
        ]

        for i, concept in enumerate(concepts):
            print(f"{i}: Concept '{concept.name}'")

        if self.config.concepts is not None:
            print(f"StoreConcepts: Concepts already present (num={len(self.config.concepts)})")

        self.config.concepts = concepts

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {}
