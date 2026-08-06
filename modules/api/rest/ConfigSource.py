import json
from pathlib import Path
from typing import Any

from modules.api.rest.ApiError import InvalidConfigError
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig

from pydantic import BaseModel, ConfigDict

DEFAULT_SECRETS_PATH = "secrets.json"


class ConfigSource(BaseModel):
    """
    Where a training run's config comes from, and how to turn it into one.

    Mirrors the arguments of scripts/train.py, and resolve() assembles them in
    the same order the CLI does -- preset, then config, then overrides -- so the
    two entry points can never disagree about what a given set of inputs means.

    Credentials are never carried here. They are read server-side from
    secrets_path; a config that tries to bring its own is rejected.
    """

    # Unknown fields are an error, so a body carrying a top-level "secrets"
    # block is told no rather than quietly ignored.
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any] | None = None
    config_path: str | None = None
    preset_path: str | None = None
    config_values: list[str] | None = None
    secrets_path: str | None = None

    def resolve(self) -> TrainConfig:
        if (self.config is None) == (self.config_path is None):
            raise InvalidConfigError("Provide exactly one of 'config' or 'config_path'")

        if self.config is not None and "secrets" in self.config:
            raise InvalidConfigError(
                "Inline secrets are not accepted. Use 'secrets_path' to point at a secrets file."
            )

        train_config = TrainConfig.default_values()

        if self.preset_path is not None:
            self.__apply_document(train_config, self.__read_json(self.preset_path, "preset"), migrate=False)

        document = self.config if self.config is not None else self.__read_json(self.config_path, "config")
        self.__apply_document(train_config, document, migrate=self.preset_path is None)

        for config_value in self.config_values or []:
            self.__apply_override(train_config, config_value)

        train_config.secrets = self.__load_secrets(self.secrets_path)
        return train_config

    @staticmethod
    def __read_json(path: str, label: str) -> dict:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise InvalidConfigError(f"{label} file not found: {path}") from None
        except json.JSONDecodeError as e:
            raise InvalidConfigError(f"{label} file is not valid JSON: {path} ({e})") from e

    @staticmethod
    def __apply_document(train_config: TrainConfig, document: dict, migrate: bool) -> None:
        try:
            train_config.from_dict(document, migrate=migrate)
        except Exception as e:
            raise InvalidConfigError(f"Invalid config: {e}") from e

    @staticmethod
    def __apply_override(train_config: TrainConfig, config_value: str) -> None:
        key, separator, value = config_value.partition("=")
        if not separator:
            raise InvalidConfigError(f"Override must be KEY=VALUE: {config_value!r}")

        if key == "secrets" or key.startswith("secrets."):
            raise InvalidConfigError(
                "Overrides may not set 'secrets'. Use 'secrets_path' to point at a secrets file."
            )

        *parent_keys, leaf_key = key.split(".")

        try:
            target = train_config
            for parent_key in parent_keys:
                target = getattr(target, parent_key)
            # Subscript, not .get(): an unknown key must raise KeyError here, or
            # from_dict silently ignores it and a typo'd override starts a run
            # with the wrong config. scripts/train.py:34 subscripts too.
            if target.types[leaf_key] is bool:
                value = value.lower() in ("true", "1", "yes")
            target.from_dict({leaf_key: value}, migrate=False)
        except Exception as e:
            raise InvalidConfigError(f"Invalid override {config_value!r}: {e}") from e

    @staticmethod
    def __load_secrets(secrets_path: str | None) -> SecretsConfig:
        explicit = secrets_path is not None
        path = Path(secrets_path or DEFAULT_SECRETS_PATH)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # Mirrors scripts/train.py: the implicit secrets.json may be absent,
            # but a path the caller named explicitly must exist.
            if explicit:
                raise InvalidConfigError(f"secrets file not found: {path}") from None
            return SecretsConfig.default_values()
        except json.JSONDecodeError as e:
            raise InvalidConfigError(f"secrets file is not valid JSON: {path} ({e})") from e
        return SecretsConfig.default_values().from_dict(document)
