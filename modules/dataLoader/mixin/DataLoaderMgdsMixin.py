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
    ):
        if config.concepts is not None:
            concepts = [concept.to_dict() for concept in config.concepts]
        else:
            with open(config.concept_file_name) as f:
                concepts = json.load(f)
                for i in range(len(concepts)):
                    concepts[i] = ConceptConfig.default_values().from_dict(concepts[i]).to_dict()

        settings = {
            "target_resolution": config.resolution,
        }

        # Just defaults for now.
        return MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size,
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

