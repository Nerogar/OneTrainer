import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        if config.concepts is not None:
            concepts = [concept.to_dict() for concept in config.concepts]
        else:
            with open(config.concept_file_name, 'r') as f:
                concepts_source = json.load(f)
            concepts = []
            for concept in concepts_source:
                if not config.validation or is_validation == concept['validation_concept']:
                    concepts.append(ConceptConfig.default_values().from_dict(concept).to_dict())

        settings = {
            "target_resolution": config.resolution,
        }

        # Just defaults for now.
        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size,
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
