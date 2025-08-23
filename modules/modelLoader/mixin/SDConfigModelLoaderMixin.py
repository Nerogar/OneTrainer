import os
from abc import ABCMeta

from modules.util.enum.ModelType import ModelType

import yaml


class SDConfigModelLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        return None

    def _get_sd_config_name(
            self,
            model_type: ModelType,
            base_model_name: str | None = None,
    ) -> str | None:
        yaml_name = None

        if base_model_name:
            new_yaml_name = os.path.splitext(base_model_name)[0] + '.yaml'
            if os.path.exists(new_yaml_name):
                yaml_name = new_yaml_name

            if not yaml_name:
                new_yaml_name = os.path.splitext(base_model_name)[0] + '.yml'
                if os.path.exists(new_yaml_name):
                    yaml_name = new_yaml_name

        if not yaml_name:
            new_yaml_name = self._default_sd_config_name(model_type)
            if new_yaml_name:
                yaml_name = new_yaml_name

        return yaml_name

    def _load_sd_config(
            self,
            model_type: ModelType,
            base_model_name: str | None = None,
    ) -> dict | None:
        yaml_name = self._get_sd_config_name(model_type, base_model_name)

        if yaml_name:
            with open(yaml_name, "r") as f:
                return yaml.safe_load(f)
        else:
            return None
