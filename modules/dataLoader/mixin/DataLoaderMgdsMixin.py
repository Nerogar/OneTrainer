import copy
import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_curation_util import is_dpo_concept_type
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):
    @staticmethod
    def __sanitize_dpo_concept(concept: ConceptConfig) -> ConceptConfig:
        sanitized = copy.deepcopy(concept)
        sanitized.image_variations = 1
        sanitized.text_variations = 1

        sanitized.image.enable_crop_jitter = False
        sanitized.image.enable_random_flip = False
        sanitized.image.enable_fixed_flip = False
        sanitized.image.enable_random_rotate = False
        sanitized.image.enable_fixed_rotate = False
        sanitized.image.random_rotate_max_angle = 0.0
        sanitized.image.enable_random_brightness = False
        sanitized.image.enable_fixed_brightness = False
        sanitized.image.random_brightness_max_strength = 0.0
        sanitized.image.enable_random_contrast = False
        sanitized.image.enable_fixed_contrast = False
        sanitized.image.random_contrast_max_strength = 0.0
        sanitized.image.enable_random_saturation = False
        sanitized.image.enable_fixed_saturation = False
        sanitized.image.random_saturation_max_strength = 0.0
        sanitized.image.enable_random_hue = False
        sanitized.image.enable_fixed_hue = False
        sanitized.image.random_hue_max_strength = 0.0
        sanitized.image.enable_random_circular_mask_shrink = False
        sanitized.image.enable_random_mask_rotate_crop = False

        sanitized.text.enable_tag_shuffling = False
        sanitized.text.tag_dropout_enable = False
        sanitized.text.tag_dropout_probability = 0.0
        sanitized.text.caps_randomize_enable = False
        sanitized.text.caps_randomize_probability = 0.0
        sanitized.text.caps_randomize_lowercase = False

        return sanitized

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

        if is_validation:
            valid_types = {ConceptType.VALIDATION, ConceptType.DPO_CHOSEN_VAL, ConceptType.DPO_REJECTED_VAL}
        else:
            valid_types = {
                ConceptType.STANDARD,
                ConceptType.PRIOR_PREDICTION,
                ConceptType.DPO_CHOSEN,
                ConceptType.DPO_REJECTED,
            }
        concepts = [
            self.__sanitize_dpo_concept(concept) if is_dpo_concept_type(ConceptType(concept.type)) else concept
            for concept in concepts
            if ConceptType(concept.type) in valid_types
        ]
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
