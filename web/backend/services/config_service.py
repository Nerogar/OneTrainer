import json
import logging
import os
import tempfile
import threading
from contextlib import suppress

import modules.util.create  # noqa: F401 — breaks circular import: optimizer_util->create->ChromaEmbeddingSetup->optimizer_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import change_optimizer
from web.backend.paths import PRESETS_DIR, SECRETS_PATH
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

_DEFAULT_PRESET_PATH = os.path.join(PRESETS_DIR, "#.json")


class ConfigService(SingletonMixin):
    _validate_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.config: TrainConfig = TrainConfig.default_values()
        self.config.from_dict(self.config.to_dict())
        self._config_lock = threading.Lock()
        self._load_default_preset()
        self._load_secrets()

    def _load_secrets(self) -> None:
        if not os.path.isfile(SECRETS_PATH):
            return
        try:
            with open(SECRETS_PATH, "r", encoding="utf-8") as fh:
                secrets_dict = json.load(fh)
            self.config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
        except Exception:
            logger.warning("Failed to load secrets from %s", SECRETS_PATH, exc_info=True)

    def _load_default_preset(self) -> None:
        if os.path.isfile(_DEFAULT_PRESET_PATH):
            try:
                with open(_DEFAULT_PRESET_PATH, "r", encoding="utf-8") as fh:
                    loaded_dict: dict = json.load(fh)
                loaded_dict["__version"] = self.config.config_version
                self.config.from_dict(loaded_dict)
                logger.info("Restored config from %s", _DEFAULT_PRESET_PATH)
                return
            except Exception:
                logger.warning(
                    "Failed to load default preset %s, attempting first-run seed",
                    _DEFAULT_PRESET_PATH,
                    exc_info=True,
                )

        seed_path = self._find_first_run_seed_preset()
        if seed_path is None:
            return

        try:
            with open(seed_path, "r", encoding="utf-8") as fh:
                loaded_dict = json.load(fh)
            loaded_dict["__version"] = self.config.config_version
            self.config.from_dict(loaded_dict)
            logger.info("First-run: seeded config from %s", seed_path)
            self._save_default_preset()
        except Exception:
            logger.warning("Failed to seed first-run config from %s", seed_path, exc_info=True)

    def _find_first_run_seed_preset(self) -> str | None:
        if not os.path.isdir(PRESETS_DIR):
            return None

        try:
            entries = sorted(os.listdir(PRESETS_DIR))
        except OSError:
            return None

        def is_builtin(name: str) -> bool:
            return name.startswith("#") and name.endswith(".json") and name != "#.json"

        builtins = [name for name in entries if is_builtin(name)]
        if not builtins:
            return None

        for name in builtins:
            lowered = name.lower()
            if "z-image" in lowered or "z_image" in lowered:
                return os.path.join(PRESETS_DIR, name)

        return os.path.join(PRESETS_DIR, builtins[0])

    def _save_default_preset(self) -> None:
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            settings_dict = self.config.to_settings_dict(secrets=False)
            fd, tmp_path = tempfile.mkstemp(dir=PRESETS_DIR, suffix=".tmp", prefix="#")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(settings_dict, fh, indent=4)
                os.replace(tmp_path, _DEFAULT_PRESET_PATH)
            except BaseException:
                with suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except Exception:
            logger.warning("Failed to save default preset %s", _DEFAULT_PRESET_PATH, exc_info=True)

    def get_config_dict(self) -> dict:
        with self._config_lock:
            return self.config.to_dict()

    def update_config(self, data: dict) -> dict:
        with self._config_lock:
            if "__version" not in data:
                data["__version"] = self.config.config_version
            # secrets are persisted exclusively through the /secrets endpoints (which
            # can hold masked placeholder values in the frontend's copy); never let
            # this general sync path overwrite the real in-memory secrets with those.
            data.pop("secrets", None)
            self.config.from_dict(data)
            result = self.config.to_dict()
            self._save_default_preset()
        return result

    def get_defaults(self) -> dict:
        return TrainConfig.default_values().to_dict()

    def load_preset(self, preset_path: str) -> dict:
        with self._config_lock:
            basename = os.path.basename(preset_path)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(preset_path, "r", encoding="utf-8") as fh:
                loaded_dict: dict = json.load(fh)

            default_config = TrainConfig.default_values()

            if is_built_in_preset:
                loaded_dict["__version"] = default_config.config_version

            loaded_config = default_config.from_dict(loaded_dict).to_unpacked_config()

            with suppress(FileNotFoundError), open(SECRETS_PATH, "r", encoding="utf-8") as fh:
                secrets_dict = json.load(fh)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.config.from_dict(loaded_config.to_dict())

            optimizer_config = change_optimizer(self.config)
            self.config.optimizer.from_dict(optimizer_config.to_dict())

            return self.config.to_dict()

    def save_preset(self, path: str) -> None:
        with self._config_lock:
            settings_dict = self.config.to_settings_dict(secrets=False)

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(settings_dict, fh, indent=4)

    def change_optimizer(self, new_optimizer: str) -> dict:
        with self._config_lock:
            from modules.util.enum.Optimizer import Optimizer
            from modules.util.optimizer_util import update_optimizer_config

            update_optimizer_config(self.config)

            new_opt_enum = Optimizer[new_optimizer]
            self.config.optimizer.optimizer = new_opt_enum

            optimizer_config = change_optimizer(self.config)
            self.config.optimizer.from_dict(optimizer_config.to_dict())

            return self.config.to_dict()

    def get_config_for_training(self) -> TrainConfig:
        with self._config_lock:
            config_dict = self.config.to_dict()

        train_config = TrainConfig.default_values()
        train_config.from_dict(config_dict)
        return train_config

    def validate_config(self, data: dict) -> dict:
        import contextlib
        import io

        validation_data = dict(data)
        if "__version" not in validation_data:
            validation_data["__version"] = TrainConfig.default_values().config_version

        errors: list[str] = []

        with self._validate_lock:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                try:
                    test_config = TrainConfig.default_values()
                    test_config.from_dict(validation_data)
                except Exception as exc:
                    errors.append(str(exc))

            output = captured.getvalue()
            for line in output.splitlines():
                line = line.strip()
                if line:
                    errors.append(line)

        if errors:
            return {"valid": False, "errors": errors}
        return {"valid": True}

    def update_cloud_secrets(self, cloud_secrets_dict: dict) -> None:
        with self._config_lock:
            self.config.secrets.cloud.from_dict(cloud_secrets_dict)

    def export_config(self) -> dict:
        with self._config_lock:
            return self.config.to_pack_dict(secrets=False)
