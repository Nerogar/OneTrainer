from modules.util.dpo_pattern_util import match_chosen, validate_dpo_patterns

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class FilterDPOChosenPaths(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Restricts collected paths of DPO pattern concepts to their chosen images.

    Rows of concepts without DPO patterns pass through untouched. For a pattern
    concept, only paths matching the chosen pattern survive — rejected-side
    images and stray files cease to exist for every downstream module, because
    this module re-emits 'image_path' and 'concept'. Runs unconditionally so a
    pattern concept never feeds its rejected images into a non-DPO run either.
    """

    def __init__(self, path_in_name: str = 'image_path', concept_in_name: str = 'concept'):
        super().__init__()

        self.path_in_name = path_in_name
        self.concept_in_name = concept_in_name

        self._kept: list[int] = []

    def length(self) -> int:
        return len(self._kept)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def start(self, variation: int):
        self._kept = []
        kept_per_pattern_concept: dict[str, int] = {}

        for index in range(self._get_previous_length(self.path_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            chosen_pattern = concept.get('dpo_chosen_pattern', '')
            if not chosen_pattern:
                self._kept.append(index)
                continue

            validate_dpo_patterns(chosen_pattern, concept.get('dpo_rejected_pattern', ''))
            if '/' in chosen_pattern.replace('\\', '/') and not concept.get('include_subdirectories', False):
                raise RuntimeError(
                    f"DPO concept '{concept.get('name') or concept.get('path')}' uses the pattern "
                    f"'{chosen_pattern}' but 'Include Subdirectories' is disabled, so its images "
                    "are never collected."
                )

            concept_path = concept['path']
            kept_per_pattern_concept.setdefault(concept_path, 0)
            image_path = self._get_previous_item(variation, self.path_in_name, index)
            if match_chosen(chosen_pattern, concept_path, image_path) is not None:
                self._kept.append(index)
                kept_per_pattern_concept[concept_path] += 1

        empty = [path for path, count in kept_per_pattern_concept.items() if count == 0]
        if empty:
            raise RuntimeError(
                "No images matched the DPO chosen pattern for: " + ", ".join(sorted(empty))
            )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        previous_index = self._kept[index]
        return {
            self.path_in_name: self._get_previous_item(variation, self.path_in_name, previous_index),
            self.concept_in_name: self._get_previous_item(variation, self.concept_in_name, previous_index),
        }
