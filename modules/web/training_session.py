import contextlib
import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from modules.modelSampler.BaseModelSampler import ModelSamplerOutput
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.FileType import FileType
from modules.util.TrainProgress import TrainProgress


def _default_progress() -> dict[str, int]:
    return {
        "epoch": 0,
        "epoch_step": 0,
        "epoch_sample": 0,
        "global_step": 0,
        "max_step": 0,
        "max_epoch": 0,
    }


@dataclass
class TrainingSession:
    train_config: TrainConfig
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    status: str = "idle"
    running: bool = False
    last_error: str | None = None
    progress: dict[str, int] = field(default_factory=_default_progress)

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _callbacks: TrainCallbacks | None = None
    _commands: TrainCommands | None = None
    _event_queue: list[dict[str, Any]] = field(default_factory=list)
    _sample_images: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.session_id,
                "created_at": self.created_at,
                "status": self.status,
                "running": self.running,
                "last_error": self.last_error,
                "progress": dict(self.progress),
                "sample_images": list(self._sample_images),
            }

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._event_queue.append(
                {
                    "type": event_type,
                    "ts": time.time(),
                    "session_id": self.session_id,
                    "payload": payload,
                }
            )

    def consume_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._event_queue)
            self._event_queue.clear()
            return events

    def _on_update_train_progress(self, train_progress: TrainProgress, max_step: int, max_epoch: int) -> None:
        payload = {
            "epoch": train_progress.epoch,
            "epoch_step": train_progress.epoch_step,
            "epoch_sample": train_progress.epoch_sample,
            "global_step": train_progress.global_step,
            "max_step": max_step,
            "max_epoch": max_epoch,
        }
        with self._lock:
            self.progress = payload
        self._append_event("progress", payload)

    def _on_update_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self._append_event("status", {"status": status})

    def _on_sample_default(self, sampler_output: ModelSamplerOutput) -> None:
        if sampler_output.file_type != FileType.IMAGE:
            return

        with contextlib.suppress(Exception):
            payload = {
                "image": self._encode_image_base64(sampler_output),
            }
            with self._lock:
                self._sample_images.append(payload["image"])
                # Keep memory bounded.
                self._sample_images = cast(list[str], self._sample_images[-20:])
            self._append_event("sample", payload)

    @staticmethod
    def _encode_image_base64(sampler_output: ModelSamplerOutput) -> str:
        import base64
        import io

        image = sampler_output.data
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Training is already running for this session")
            self.last_error = None
            self.running = True
            self.status = "starting"

        self._append_event("status", {"status": "starting"})
        self._commands = TrainCommands()
        self._callbacks = TrainCallbacks(
            on_update_train_progress=self._on_update_train_progress,
            on_update_status=self._on_update_status,
            on_sample_default=self._on_sample_default,
            on_sample_custom=self._on_sample_default,
        )

        thread = threading.Thread(target=self._training_thread_function, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        commands = self._commands
        if commands:
            commands.stop()
            self._append_event("status", {"status": "stopping"})

    def sample_default(self) -> None:
        commands = self._commands
        if commands:
            commands.sample_default()

    def backup(self) -> None:
        commands = self._commands
        if commands:
            commands.backup()

    def save(self) -> None:
        commands = self._commands
        if commands:
            commands.save()

    def _training_thread_function(self) -> None:
        error_caught = False
        error_text: str | None = None

        callbacks = self._callbacks
        commands = self._commands
        if callbacks is None or commands is None:
            with self._lock:
                self.running = False
                self.status = "error"
                self.last_error = "Internal error: callbacks/commands not initialized"
                self._thread = None
            return

        trainer = create.create_trainer(self.train_config, callbacks, commands)
        try:
            trainer.start()
            trainer.train()
        except Exception:
            error_caught = True
            error_text = traceback.format_exc()
        finally:
            with contextlib.suppress(Exception):
                trainer.end()

        with self._lock:
            self.running = False
            self._thread = None
            self._callbacks = None
            self._commands = None

            if error_caught:
                self.status = "error"
                self.last_error = error_text
            else:
                self.status = "stopped"
                self.last_error = None

        if error_caught:
            self._append_event("error", {"error": error_text})
        else:
            self._append_event("status", {"status": "stopped"})


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, TrainingSession] = {}

    def create_session(
        self,
        config_payload: dict[str, Any] | None = None,
        secrets_payload: dict[str, Any] | None = None,
    ) -> TrainingSession:
        train_config = TrainConfig.default_values()
        if config_payload:
            train_config.from_dict(config_payload)

        if secrets_payload:
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_payload)

        session = TrainingSession(train_config=train_config)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def list_sessions(self) -> list[TrainingSession]:
        with self._lock:
            return list(self._sessions.values())

    def get_session(self, session_id: str) -> TrainingSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session id: {session_id}")
        return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.running:
                raise RuntimeError("Cannot delete a running session")
            self._sessions.pop(session_id, None)

    def export_config_json(self, session_id: str) -> str:
        session = self.get_session(session_id)
        packed = session.train_config.to_pack_dict(secrets=False)
        return json.dumps(packed, indent=2)
