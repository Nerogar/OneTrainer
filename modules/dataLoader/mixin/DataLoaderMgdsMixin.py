import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_pattern_util import validate_dpo_patterns
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):
    @staticmethod
    def __filter_dpo_concepts(concepts: list[ConceptConfig]) -> list[ConceptConfig]:
        # An RLHF DPO run trains on chosen/rejected pairs only. Concepts without
        # patterns have no pair data, so they are dropped (previously they were
        # dropped silently at pair-matching time).
        for concept in concepts:
            if concept.enabled:
                validate_dpo_patterns(concept.dpo_chosen_pattern, concept.dpo_rejected_pattern)

        dpo_concepts = [c for c in concepts if c.is_dpo()]
        skipped = [c.name or c.path for c in concepts if c.enabled and not c.is_dpo()]

        if not any(c.enabled for c in dpo_concepts):
            raise RuntimeError(
                "RLHF DPO requires at least one enabled concept with chosen/rejected patterns "
                "(set 'DPO Chosen Pattern' and 'DPO Rejected Pattern' in the concept window)."
            )
        if skipped:
            print(f"RLHF DPO: skipping {len(skipped)} concepts without DPO patterns: " + ", ".join(skipped))

        return dpo_concepts

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

        valid_types = {ConceptType.VALIDATION} if is_validation else {ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION}
        concepts = [concept for concept in concepts if ConceptType(concept.type) in valid_types]
        if config.rlhf_enabled:
            concepts = self.__filter_dpo_concepts(concepts)
        concepts = [c.to_dict() for c in concepts]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
