from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class SecretsConfig(BaseConfig):
    huggingface_token: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values() -> 'SecretsConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("huggingface_token", "", str, False))

        return SecretsConfig(data)
