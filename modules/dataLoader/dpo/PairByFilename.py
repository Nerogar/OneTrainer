import os

from modules.util.dpo_curation_util import dpo_pair_key

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class PairByFilename(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
        self,
        concept_pairs: list[tuple[str, str]],
        chosen_names: list[str | tuple[str, str]],
        rejected_names: list[str | tuple[str, str]],
    ):
        super().__init__()

        self.chosen_names = [x if isinstance(x, tuple) else (x, x) for x in chosen_names]
        self.rejected_names = [x if isinstance(x, tuple) else (x, x) for x in rejected_names]

        self.concept_lookup = {}
        for pair_id, (chosen_path, rejected_path) in enumerate(concept_pairs):
            self.concept_lookup[self.__canonical_path(chosen_path)] = (pair_id, True)
            self.concept_lookup[self.__canonical_path(rejected_path)] = (pair_id, False)

        self._pair_indices: list[tuple[int, int]] | None = None

    @staticmethod
    def __canonical_path(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def __build_pair_indices(self):
        chosen_indices = {}
        rejected_indices = {}

        for index in range(self._get_previous_length("image_path")):
            concept_path = self.__canonical_path(self._get_previous_item(0, "concept.path", index))
            pair_info = self.concept_lookup.get(concept_path)
            if pair_info is None:
                continue

            pair_id, is_chosen = pair_info
            image_path = self._get_previous_item(0, "image_path", index)
            key = (pair_id, dpo_pair_key(image_path, concept_path))

            if is_chosen:
                chosen_indices[key] = index
            else:
                rejected_indices[key] = index

        pair_indices = []
        missing_rejected = sorted(set(chosen_indices) - set(rejected_indices))
        missing_chosen = sorted(set(rejected_indices) - set(chosen_indices))
        if missing_rejected or missing_chosen:
            details = []
            if missing_rejected:
                details.append(f"{len(missing_rejected)} chosen files are missing rejected matches")
            if missing_chosen:
                details.append(f"{len(missing_chosen)} rejected files are missing chosen matches")
            raise RuntimeError("RLHF DPO concept pairs must match exactly by filename: " + ", ".join(details) + ".")

        for key, chosen_index in chosen_indices.items():
            rejected_index = rejected_indices.get(key)
            if rejected_index is not None:
                pair_indices.append((chosen_index, rejected_index))

        pair_indices.sort(key=lambda x: x[0])
        if not pair_indices:
            raise RuntimeError(
                "No DPO pairs could be matched by filename between the configured chosen/rejected concepts."
            )

        self._pair_indices = pair_indices

    def __get_pair_indices(self) -> list[tuple[int, int]]:
        if self._pair_indices is None:
            self.__build_pair_indices()
        return self._pair_indices

    def length(self) -> int:
        return len(self.__get_pair_indices())

    def get_inputs(self) -> list[str]:
        names = ["concept.path", "image_path", "prompt", "crop_resolution"]
        names += [in_name for in_name, _ in self.chosen_names]
        names += [in_name for in_name, _ in self.rejected_names]
        return list(dict.fromkeys(names))

    def get_outputs(self) -> list[str]:
        return [out_name for _, out_name in self.chosen_names] + [out_name for _, out_name in self.rejected_names]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        chosen_index, rejected_index = self.__get_pair_indices()[index]

        chosen_prompt = self._get_previous_item(variation, "prompt", chosen_index)
        rejected_prompt = self._get_previous_item(variation, "prompt", rejected_index)
        if chosen_prompt != rejected_prompt:
            raise RuntimeError(
                "RLHF DPO paired samples must use identical prompts/captions in chosen and rejected concepts."
            )

        chosen_crop_resolution = self._get_previous_item(variation, "crop_resolution", chosen_index)
        rejected_crop_resolution = self._get_previous_item(variation, "crop_resolution", rejected_index)
        if isinstance(chosen_crop_resolution, torch.Tensor) and isinstance(rejected_crop_resolution, torch.Tensor):
            same_resolution = torch.equal(chosen_crop_resolution, rejected_crop_resolution)
        else:
            same_resolution = chosen_crop_resolution == rejected_crop_resolution
        if not same_resolution:
            raise RuntimeError(
                "RLHF DPO paired samples must have matching crop resolutions in chosen and rejected concepts."
            )

        item = {}
        for in_name, out_name in self.chosen_names:
            item[out_name] = self._get_previous_item(variation, in_name, chosen_index)
        for in_name, out_name in self.rejected_names:
            item[out_name] = self._get_previous_item(variation, in_name, rejected_index)

        return item
