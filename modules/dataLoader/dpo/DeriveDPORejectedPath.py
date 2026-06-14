import os

from modules.util.dpo_curation_util import resolve_aspect_ratio
from modules.util.dpo_pattern_util import build_rejected_index, match_chosen, resolve_rejected

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DeriveDPORejectedPath(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Derives the rejected image path for every chosen image of a DPO pattern
    concept. Rows whose concept has no patterns emit '' (never None, which mgds
    treats as a fall-through to earlier modules). Missing rejected images are
    collected and raised as one error; aspect-ratio mismatches only warn,
    because the rejected image is scale-cropped to the chosen image's bucket.
    """

    def __init__(
            self,
            path_in_name: str = 'image_path',
            concept_in_name: str = 'concept',
            rejected_path_out_name: str = 'image_path_rejected',
    ):
        super().__init__()

        self.path_in_name = path_in_name
        self.concept_in_name = concept_in_name
        self.rejected_path_out_name = rejected_path_out_name

        self._rejected_paths: list[str] = []

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.rejected_path_out_name]

    def start(self, variation: int):
        self._rejected_paths = []
        index_cache: dict[str, dict[str, list[str]]] = {}
        missing: list[str] = []
        aspect_mismatches: list[str] = []

        for index in range(self._get_previous_length(self.path_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            rejected_pattern = concept.get('dpo_rejected_pattern', '')
            if not rejected_pattern:
                self._rejected_paths.append('')
                continue

            concept_path = concept['path']
            image_path = self._get_previous_item(variation, self.path_in_name, index)
            stem = match_chosen(concept['dpo_chosen_pattern'], concept_path, image_path)

            if concept_path not in index_cache:
                index_cache[concept_path] = build_rejected_index(
                    concept_path, concept.get('include_subdirectories', False)
                )

            try:
                rejected_path = resolve_rejected(
                    rejected_pattern, concept_path, stem,
                    os.path.splitext(image_path)[1], index_cache[concept_path],
                )
            except FileNotFoundError as e:
                missing.append(str(e))
                self._rejected_paths.append('')
                continue

            self._rejected_paths.append(rejected_path)

            if resolve_aspect_ratio("", image_path) != resolve_aspect_ratio("", rejected_path):
                aspect_mismatches.append(f"{image_path} vs {rejected_path}")

        if missing:
            raise RuntimeError(
                f"RLHF DPO: {len(missing)} chosen images have no rejected match. First errors: "
                + " | ".join(missing[:10])
            )

        if aspect_mismatches:
            print(
                f"WARNING: {len(aspect_mismatches)} DPO pairs land in different aspect buckets; "
                "the rejected image will be scaled and cropped to the chosen image's bucket. "
                "First pairs: " + " | ".join(aspect_mismatches[:10])
            )

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {
            self.rejected_path_out_name: self._rejected_paths[index],
        }
