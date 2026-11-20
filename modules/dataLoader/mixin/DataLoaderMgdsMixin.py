import json
import os
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):

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

        # choose all validation concepts, or none of them, depending on is_validation
        concepts = [concept for concept in concepts if (ConceptType(concept.type) == ConceptType.VALIDATION) == is_validation]

        if is_validation:
            labels = {}
            for concept in concepts:
                if not concept.enabled:
                    continue
                label = concept.name or os.path.basename(concept.path)
                description = f"{concept.name!r} ({concept.path!r})"
                if not label.strip() or label == "total_average":
                    raise ValueError(
                        f"Validation concept {description} has an empty or reserved label {label!r}. "
                        "Assign a distinct name other than 'total_average'.")
                if label in labels:
                    raise ValueError(
                        f"Validation concepts {labels[label]} and {description} share label {label!r}. "
                        "Assign distinct names.")
                labels[label] = description

        # convert before passing to MGDS
        concepts = [c.to_dict() for c in concepts]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        # Just defaults for now.
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
