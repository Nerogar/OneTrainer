import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig


class WebUIService:
    def __init__(self):
        self._lock = threading.Lock()
        self._train_thread: threading.Thread | None = None
        self._commands: TrainCommands | None = None
        self._trainer = None

        self._running = False
        self._status = "idle"
        self._error: str | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._last_config_path: str | None = None
        self._last_secrets_path: str | None = None
        self._progress = {
            "epoch": 0,
            "epoch_step": 0,
            "epoch_sample": 0,
            "global_step": 0,
            "max_step": 0,
            "max_epoch": 0,
        }
        self._logs: list[str] = []

    def _append_log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self._logs.append(log_entry)
        while len(self._logs) > 500:
            self._logs.pop(0)

    def _on_update_status(self, status: str):
        with self._lock:
            self._status = status
            self._append_log(status)

    def _on_update_train_progress(self, train_progress, max_step: int, max_epoch: int):
        with self._lock:
            self._progress = {
                "epoch": train_progress.epoch,
                "epoch_step": train_progress.epoch_step,
                "epoch_sample": train_progress.epoch_sample,
                "global_step": train_progress.global_step,
                "max_step": max_step,
                "max_epoch": max_epoch,
            }

    def _load_train_config(self, config_path: str, secrets_path: str | None) -> TrainConfig:
        train_config = TrainConfig.default_values()
        with open(config_path, "r", encoding="utf-8") as f:
            train_config.from_dict(json.load(f))

        if secrets_path is not None:
            with open(secrets_path, "r", encoding="utf-8") as f:
                secrets_dict = json.load(f)
                train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
        else:
            try:
                with open("secrets.json", "r", encoding="utf-8") as f:
                    secrets_dict = json.load(f)
                    train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
            except FileNotFoundError:
                pass

        return train_config

    def _train_worker(self, config_path: str, secrets_path: str | None):
        with self._lock:
            self._running = True
            self._status = "initializing"
            self._error = None
            self._started_at = time.time()
            self._ended_at = None
            self._last_config_path = str(Path(config_path).absolute())
            self._last_secrets_path = str(Path(secrets_path).absolute()) if secrets_path else None
            self._progress = {
                "epoch": 0,
                "epoch_step": 0,
                "epoch_sample": 0,
                "global_step": 0,
                "max_step": 0,
                "max_epoch": 0,
            }
            self._logs = []
            self._append_log("Training requested from WebUI")

        callbacks = TrainCallbacks(
            on_update_train_progress=self._on_update_train_progress,
            on_update_status=self._on_update_status,
        )
        commands = TrainCommands()

        trainer = None
        try:
            train_config = self._load_train_config(config_path, secrets_path)
            trainer = create.create_trainer(train_config, callbacks, commands)

            with self._lock:
                self._commands = commands
                self._trainer = trainer

            trainer.start()
            trainer.train()
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                self._error = str(e)
                self._status = "error"
                self._append_log(f"Error: {e}")
        finally:
            if trainer is not None:
                try:
                    trainer.end()
                except Exception:
                    traceback.print_exc()

            with self._lock:
                self._commands = None
                self._trainer = None
                self._train_thread = None
                self._running = False
                self._ended_at = time.time()
                if self._status != "error":
                    self._status = "stopped"
                    self._append_log("Training finished")

    def start_training(self, config_path: str, secrets_path: str | None = None):
        with self._lock:
            if self._running:
                return False, "training is already running"

            thread = threading.Thread(
                target=self._train_worker,
                args=(config_path, secrets_path),
                daemon=True,
            )
            self._train_thread = thread
            thread.start()

        return True, "training started"

    def stop_training(self):
        with self._lock:
            if not self._running or self._commands is None:
                return False, "no running training"

            self._commands.stop()
            self._append_log("Stop command requested")

        return True, "stop requested"

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "status": self._status,
                "error": self._error,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
                "last_config_path": self._last_config_path,
                "last_secrets_path": self._last_secrets_path,
                "progress": dict(self._progress),
                "logs": list(self._logs),
            }
