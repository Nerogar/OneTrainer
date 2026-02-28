"""
Thread-safe shared state for the Gradio WebUI.

Holds training progress, status, commands, config, and sample images
so the background training thread and the Gradio polling timer can
communicate without race conditions.
"""

import threading
import time
from dataclasses import dataclass, field

from modules.util.config.TrainConfig import TrainConfig


@dataclass
class WebUIState:
    # ── training lifecycle ──────────────────────────────────────────
    running: bool = False
    stopping: bool = False
    error: bool = False

    # ── progress ────────────────────────────────────────────────────
    status: str = "idle"
    step: int = 0
    max_step: int = 0
    epoch: int = 0
    max_epoch: int = 0
    start_time: float = 0.0

    # ── config ──────────────────────────────────────────────────────
    train_config: TrainConfig = field(default_factory=TrainConfig.default_values)

    # ── commands (set when training starts, None otherwise) ─────────
    commands: object = None          # TrainCommands | None
    training_thread: object = None   # threading.Thread | None

    # ── sample images (paths written by callbacks) ──────────────────
    sample_image_paths: list = field(default_factory=list)

    # ── thread safety ───────────────────────────────────────────────
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── helpers ─────────────────────────────────────────────────────
    def calculate_eta(self) -> str:
        """Return a human-readable ETA string, or empty if unavailable."""
        if not self.running or self.max_step == 0 or self.step == 0:
            return ""
        elapsed = time.monotonic() - self.start_time
        if elapsed <= 0:
            return ""
        steps_per_sec = self.step / elapsed
        remaining_steps = self.max_step - self.step
        if steps_per_sec <= 0:
            return ""
        remaining_secs = remaining_steps / steps_per_sec

        hours, rem = divmod(int(remaining_secs), 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            return f"ETA: {hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"ETA: {minutes}m {seconds}s"
        return f"ETA: {seconds}s"

    def reset(self):
        """Reset all transient state after a training run ends."""
        self.running = False
        self.stopping = False
        self.error = False
        self.status = "idle"
        self.step = 0
        self.max_step = 0
        self.epoch = 0
        self.max_epoch = 0
        self.start_time = 0.0
        self.commands = None
        self.training_thread = None
        self.sample_image_paths.clear()


# Module-level singleton
_state: WebUIState | None = None
_state_lock = threading.Lock()


def get_state() -> WebUIState:
    """Return the singleton WebUIState, creating it on first call."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = WebUIState()
    return _state
