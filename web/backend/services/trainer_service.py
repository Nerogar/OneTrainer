import logging
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from web.backend.services._serialization import serialize_sample
from web.backend.services._singleton import SingletonMixin
from web.backend.services.concept_service import ConceptService
from web.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)

TrainingStatus = Literal["idle", "starting", "running", "stopping", "error"]


class TrainerService(SingletonMixin):
    def __init__(self) -> None:
        self._status: TrainingStatus = "idle"
        self._error_message: str | None = None
        self._training_thread: threading.Thread | None = None
        self._training_commands: Any | None = None
        self._training_callbacks: Any | None = None
        self._start_time: float | None = None
        self._status_lock = threading.Lock()
        self._ws_broadcast: Callable[[dict], None] | None = None

    def set_ws_broadcast(self, fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = fn

    def _broadcast(self, message: dict) -> None:
        if self._ws_broadcast is not None:
            with suppress(Exception):
                self._ws_broadcast(message)

    def _set_status(self, status: TrainingStatus, error_message: str | None = None) -> None:
        with self._status_lock:
            self._status = status
            self._error_message = error_message

    def get_status(self) -> dict:
        with self._status_lock:
            return {
                "status": self._status,
                "error": self._error_message,
                "start_time": self._start_time,
            }

    def start_training(self, reattach: bool = False, train_config: Any | None = None) -> dict:
        with self._status_lock:
            if self._status in ("running", "stopping", "starting"):
                return {"ok": False, "error": f"Training is already {self._status}"}
            self._status = "starting"

        config_service = ConfigService.get_instance()
        if train_config is None:
            train_config = config_service.get_config_for_training()

            concept_service = ConceptService()

            with suppress(Exception):
                concepts = concept_service.load_concepts(train_config.concept_file_name)
                concept_service.save_concepts(train_config.concept_file_name, concepts)
            with suppress(Exception):
                samples = concept_service.load_samples(train_config.sample_definition_file_name)
                concept_service.save_samples(train_config.sample_definition_file_name, samples)

        if train_config.tensorboard and not train_config.tensorboard_always_on:
            self.stop_always_on_tensorboard()

        from modules.util.callbacks.TrainCallbacks import TrainCallbacks
        from modules.util.commands.TrainCommands import TrainCommands

        commands = TrainCommands()

        callbacks = TrainCallbacks(
            on_update_train_progress=self._on_update_train_progress,
            on_update_status=self._on_update_status,
            on_sample_default=self._on_sample_default,
            on_update_sample_default_progress=self._on_update_sample_default_progress,
            on_sample_custom=self._on_sample_custom,
            on_update_sample_custom_progress=self._on_update_sample_custom_progress,
        )

        from modules.util import create

        try:
            trainer = create.create_trainer(train_config, callbacks, commands, reattach=reattach)
        except Exception as e:
            with self._status_lock:
                self._status = "idle"
            return {"ok": False, "error": str(e)}

        with suppress(Exception):
            from modules.util.torch_util import torch_gc

            torch_gc()

        self._training_commands = commands
        self._training_callbacks = callbacks
        with self._status_lock:
            self._status = "running"
        self._broadcast({"type": "status", "data": {"text": "Starting training..."}})

        thread = threading.Thread(
            target=self._training_thread_fn,
            args=(trainer, train_config),
            daemon=True,
            name="OneTrainerWeb-training",
        )
        self._training_thread = thread
        thread.start()

        return {"ok": True}

    def _training_thread_fn(self, trainer: Any, config: Any) -> None:
        error_caught = False

        try:
            trainer.start()

            if config.cloud.enabled:
                with suppress(Exception):
                    from web.backend.services.config_service import ConfigService as _CS

                    _CS.get_instance().update_cloud_secrets(config.secrets.cloud.to_dict())

            with self._status_lock:
                self._start_time = time.time()
            trainer.train()
        except Exception:
            if config.cloud.enabled:
                with suppress(Exception):
                    from web.backend.services.config_service import ConfigService as _CS

                    _CS.get_instance().update_cloud_secrets(config.secrets.cloud.to_dict())

            error_caught = True
            traceback.print_exc()
        finally:
            with suppress(Exception):
                trainer.end()

        del trainer

        with self._status_lock:
            self._training_thread = None
            self._training_commands = None
            self._training_callbacks = None

        with suppress(Exception):
            import torch

            torch.clear_autocast_cache()
        with suppress(Exception):
            from modules.util.torch_util import torch_gc

            torch_gc()

        if error_caught:
            with self._status_lock:
                self._status = "error"
                self._error_message = "Training failed -- check the Terminal panel for details"
                self._start_time = None
            self._broadcast({"type": "status", "data": {"text": "Error: check the Terminal panel for details"}})
        else:
            with self._status_lock:
                self._status = "idle"
                self._error_message = None
                self._start_time = None
            self._broadcast({"type": "status", "data": {"text": "Stopped"}})

        if config.tensorboard_always_on:
            with suppress(Exception):
                self._start_always_on_tensorboard()

    def stop_training(self) -> dict:
        with self._status_lock:
            if self._status != "running":
                return {"ok": False, "error": "Training is not running"}
            self._status = "stopping"
            self._error_message = None
            commands = self._training_commands

        self._broadcast({"type": "status", "data": {"text": "Stopping..."}})

        if commands is not None:
            with suppress(Exception):
                commands.stop()

        return {"ok": True}

    def sample_now(self) -> dict:
        with self._status_lock:
            commands = self._training_commands
        if commands is None:
            return {"ok": False, "error": "Training is not running"}
        with suppress(Exception):
            commands.sample_default()
        return {"ok": True}

    def sample_custom(self, sample_params: Any) -> dict:
        with self._status_lock:
            commands = self._training_commands
        if commands is None:
            return {"ok": False, "error": "Training is not running"}
        with suppress(Exception):
            commands.sample_custom(sample_params)
        return {"ok": True}

    def backup_now(self) -> dict:
        with self._status_lock:
            commands = self._training_commands
        if commands is None:
            return {"ok": False, "error": "Training is not running"}
        with suppress(Exception):
            commands.backup()
        return {"ok": True}

    def save_now(self) -> dict:
        with self._status_lock:
            commands = self._training_commands
        if commands is None:
            return {"ok": False, "error": "Training is not running"}
        with suppress(Exception):
            commands.save()
        return {"ok": True}

    def _on_update_train_progress(self, train_progress: Any, max_step: int, max_epoch: int) -> None:
        self._broadcast({
            "type": "progress",
            "data": {
                "epoch": train_progress.epoch,
                "epoch_step": train_progress.epoch_step,
                "epoch_sample": train_progress.epoch_sample,
                "global_step": train_progress.global_step,
                "max_step": max_step,
                "max_epoch": max_epoch,
            },
        })

    def _on_update_status(self, status_text: str) -> None:
        self._broadcast({"type": "status", "data": {"text": status_text}})

    def _on_sample_default(self, sampler_output: Any) -> None:
        with suppress(Exception):
            payload = serialize_sample(sampler_output)
            self._broadcast({"type": "sample", "data": payload})

    def _on_update_sample_default_progress(self, step: int, max_step: int) -> None:
        self._broadcast({
            "type": "sample_progress",
            "data": {"step": step, "max_step": max_step},
        })

    def _on_sample_custom(self, sampler_output: Any) -> None:
        with suppress(Exception):
            payload = serialize_sample(sampler_output)
            self._broadcast({"type": "sample", "data": payload})

    def _on_update_sample_custom_progress(self, step: int, max_step: int) -> None:
        self._broadcast({
            "type": "sample_progress",
            "data": {"step": step, "max_step": max_step},
        })

    _always_on_tensorboard_subprocess: Any = None

    def _start_always_on_tensorboard(self) -> None:
        import os
        import subprocess
        import sys

        self.stop_always_on_tensorboard()

        config = ConfigService.get_instance().config

        tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(os.path.abspath(tensorboard_log_dir), exist_ok=True)

        args = [
            tensorboard_executable,
            "--logdir", tensorboard_log_dir,
            "--port", str(config.tensorboard_port),
            "--samples_per_plugin=images=100,scalars=10000",
        ]

        if config.tensorboard_expose:
            args.append("--bind_all")

        try:
            self.__class__._always_on_tensorboard_subprocess = subprocess.Popen(args)
        except Exception:
            self.__class__._always_on_tensorboard_subprocess = None

    def stop_always_on_tensorboard(self) -> None:
        import subprocess

        proc = self.__class__._always_on_tensorboard_subprocess
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
            finally:
                self.__class__._always_on_tensorboard_subprocess = None
