import random
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType


class ConceptImageConfig(BaseConfig):
    enable_crop_jitter: bool

    enable_random_flip: bool
    enable_fixed_flip: bool

    enable_random_rotate: bool
    enable_fixed_rotate: bool
    random_rotate_max_angle: float

    enable_random_brightness: bool
    enable_fixed_brightness: bool
    random_brightness_max_strength: float

    enable_random_contrast: bool
    enable_fixed_contrast: bool
    random_contrast_max_strength: float

    enable_random_saturation: bool
    enable_fixed_saturation: bool
    random_saturation_max_strength: float

    enable_random_hue: bool
    enable_fixed_hue: bool
    random_hue_max_strength: float

    enable_resolution_override: bool
    resolution_override: str

    enable_random_circular_mask_shrink: bool

    enable_random_mask_rotate_crop: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("enable_crop_jitter", True, bool, False))

        data.append(("enable_random_flip", False, bool, False))
        data.append(("enable_fixed_flip", False, bool, False))

        data.append(("enable_random_rotate", False, bool, False))
        data.append(("enable_fixed_rotate", False, bool, False))
        data.append(("random_rotate_max_angle", 0.0, float, False))

        data.append(("enable_random_brightness", False, bool, False))
        data.append(("enable_fixed_brightness", False, bool, False))
        data.append(("random_brightness_max_strength", 0.0, float, False))

        data.append(("enable_random_contrast", False, bool, False))
        data.append(("enable_fixed_contrast", False, bool, False))
        data.append(("random_contrast_max_strength", 0.0, float, False))

        data.append(("enable_random_saturation", False, bool, False))
        data.append(("enable_fixed_saturation", False, bool, False))
        data.append(("random_saturation_max_strength", 0.0, float, False))

        data.append(("enable_random_hue", False, bool, False))
        data.append(("enable_fixed_hue", False, bool, False))
        data.append(("random_hue_max_strength", 0.0, float, False))

        data.append(("enable_resolution_override", False, bool, False))
        data.append(("resolution_override", "512", str, False))

        data.append(("enable_random_circular_mask_shrink", False, bool, False))

        data.append(("enable_random_mask_rotate_crop", False, bool, False))

        return ConceptImageConfig(data)


class ConceptTextConfig(BaseConfig):
    prompt_source: str
    prompt_path: str
    enable_tag_shuffling: bool
    tag_delimiter: str
    keep_tags_count: int
    tag_dropout_enable: bool
    tag_dropout_mode: str
    tag_dropout_probability: float
    tag_dropout_special_tags_mode: str
    tag_dropout_special_tags: str
    tag_dropout_special_tags_regex: bool
    caps_randomize_enable: bool
    caps_randomize_probability: float
    caps_randomize_mode: str
    caps_randomize_lowercase: bool


    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("prompt_source", "sample", str, False))
        data.append(("prompt_path", "", str, False))
        data.append(("enable_tag_shuffling", False, bool, False))
        data.append(("tag_delimiter", ",", str, False))
        data.append(("keep_tags_count", 1, int, False))
        data.append(("tag_dropout_enable", False, bool, False))
        data.append(("tag_dropout_mode", "FULL", str, False))
        data.append(("tag_dropout_probability", 0.0, float, False))
        data.append(("tag_dropout_special_tags_mode", "NONE", str, False))
        data.append(("tag_dropout_special_tags", "", str, False))
        data.append(("tag_dropout_special_tags_regex", False, bool, False))
        data.append(("caps_randomize_enable", False, bool, False))
        data.append(("caps_randomize_mode", "capslock, title, first, random", str, False))
        data.append(("caps_randomize_probability", 0.0, float, False))
        data.append(("caps_randomize_lowercase", False, bool, False))

        return ConceptTextConfig(data)


class ConceptConfig(BaseConfig):
    name: str
    path: str
    seed: int
    enabled: bool
    type: ConceptType
    include_subdirectories: bool
    image_variations: int
    text_variations: int
    repeats: float
    loss_weight: float
    concept_stats: dict

    image: ConceptImageConfig
    text: ConceptTextConfig

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(
            data,
            config_version=2,
            config_migrations={
                0: self.__migration_0,
                1: self.__migration_1,
            }
        )

    def __migration_0(self, data: dict) -> dict:
        migrated_data = {}
        for key, value in data.items():
            if key == 'repeats':
                migrated_data['balancing'] = value
            else:
                migrated_data[key] = value

        return migrated_data

    def __migration_1(self, data: dict) -> dict:
        migrated_data = {}
        for key, value in data.items():
            if key == 'validation_concept':
                migrated_data['type'] = ConceptType.VALIDATION if value else ConceptType.STANDARD
            else:
                migrated_data[key] = value

        return migrated_data

    def to_dict(self):
        as_dict = super().to_dict()
        as_dict['image'] = self.image.to_dict()
        as_dict['text'] = self.text.to_dict()
        return as_dict

    @staticmethod
    def default_values():
        data = []

        data.append(("image", ConceptImageConfig.default_values(), ConceptImageConfig, False))
        data.append(("text", ConceptTextConfig.default_values(), ConceptTextConfig, False))

        data.append(("name", "", str, False))
        data.append(("path", "", str, False))
        data.append(("seed", random.randint(-(1 << 30), 1 << 30), int, False))
        data.append(("enabled", True, bool, False))
        data.append(("type", ConceptType.STANDARD, ConceptType, False))
        data.append(("include_subdirectories", False, bool, False))
        data.append(("image_variations", 1, int, False))
        data.append(("text_variations", 1, int, False))
        data.append(("balancing", 1.0, float, False))
        data.append(("balancing_strategy", BalancingStrategy.REPEATS, BalancingStrategy, False))
        data.append(("loss_weight", 1.0, float, False))
        data.append(("concept_stats", {}, dict, False))

        return ConceptConfig(data)
