import json
import logging
import os
from typing import Any

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig

logger = logging.getLogger(__name__)


class ConceptService:
    def _load_list(self, file_path: str, config_class: Any) -> list[dict]:
        with open(file_path, "r", encoding="utf-8") as fh:
            raw_list: list[dict] = json.load(fh)

        return [config_class.default_values().from_dict(entry).to_dict() for entry in raw_list]

    def _save_list(self, file_path: str, items: list[dict], config_class: Any) -> None:
        normalised = [config_class.default_values().from_dict(entry).to_dict() for entry in items]

        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(normalised, fh, indent=4)

    def load_concepts(self, file_path: str) -> list[dict]:
        return self._load_list(file_path, ConceptConfig)

    def save_concepts(self, file_path: str, concepts: list[dict]) -> None:
        self._save_list(file_path, concepts, ConceptConfig)

    def load_samples(self, file_path: str) -> list[dict]:
        return self._load_list(file_path, SampleConfig)

    def save_samples(self, file_path: str, samples: list[dict]) -> None:
        self._save_list(file_path, samples, SampleConfig)
