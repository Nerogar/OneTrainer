"""
Bridge between the Gradio WebUI and the OneTrainer training backend.

Provides functions to start/stop training in a background thread,
issue runtime commands (sample, backup, save), and poll progress
for the gr.Timer callback.
"""

import threading
import time
import traceback

from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.torch_util import torch_gc
from modules.webui.webui_state import WebUIState

import torch


# ── callbacks ───────────────────────────────────────────────────────────────

def _make_callbacks(state: WebUIState) -> TrainCallbacks:
    """Create TrainCallbacks that write into the shared state."""

    def on_progress(train_progress, max_step, max_epoch):
        with state.lock:
            state.step = train_progress.global_step
            state.max_step = max_step
            state.epoch = train_progress.epoch
            state.max_epoch = max_epoch

    def on_status(status_text):
        with state.lock:
            state.status = status_text

    return TrainCallbacks(
        on_update_train_progress=on_progress,
        on_update_status=on_status,
    )


# ── training thread ────────────────────────────────────────────────────────

def _training_thread(state: WebUIState):
    """Target for the background training thread."""
    error_caught = False
    try:
        callbacks = _make_callbacks(state)
        trainer = create.create_trainer(
            state.train_config, callbacks, state.commands,
        )
        trainer.start()
        with state.lock:
            state.start_time = time.monotonic()
        trainer.train()
    except Exception:
        error_caught = True
        traceback.print_exc()
    finally:
        try:
            trainer.end()
        except Exception:
            traceback.print_exc()

        del trainer
        torch.clear_autocast_cache()
        torch_gc()

        with state.lock:
            state.running = False
            state.stopping = False
            state.error = error_caught
            state.status = (
                "error — check console" if error_caught else "finished"
            )
            state.commands = None
            state.training_thread = None


# ── public API ──────────────────────────────────────────────────────────────

def start_training(state: WebUIState) -> str:
    """
    Launch training in a background thread.
    Returns a status string for immediate UI feedback.
    """
    with state.lock:
        if state.running:
            return "already running"
        state.running = True
        state.stopping = False
        state.error = False
        state.status = "starting…"
        state.step = 0
        state.max_step = 0
        state.epoch = 0
        state.max_epoch = 0
        state.commands = TrainCommands()

    t = threading.Thread(target=_training_thread, args=(state,), daemon=True)
    with state.lock:
        state.training_thread = t
    t.start()
    return "training started"


def stop_training(state: WebUIState) -> str:
    """Request a graceful stop of the running training."""
    with state.lock:
        if not state.running:
            return "not running"
        if state.stopping:
            return "already stopping"
        state.stopping = True
        state.status = "stopping…"
        if state.commands is not None:
            state.commands.stop()
    return "stop requested"


def sample_now(state: WebUIState):
    with state.lock:
        if state.commands is not None:
            state.commands.sample_default()


def backup_now(state: WebUIState):
    with state.lock:
        if state.commands is not None:
            state.commands.backup()


def save_now(state: WebUIState):
    with state.lock:
        if state.commands is not None:
            state.commands.save()


# ── polling ─────────────────────────────────────────────────────────────────

def poll_state(state: WebUIState) -> dict:
    """
    Snapshot the training state for the gr.Timer callback.
    Returns a dict of values to push into Gradio components.
    """
    with state.lock:
        progress = (
            state.step / state.max_step if state.max_step > 0 else 0.0
        )
        running = state.running
        stopping = state.stopping

        if not running:
            btn_label = "start training"
            btn_variant = "primary"
            btn_interactive = True
        elif stopping:
            btn_label = "stopping…"
            btn_variant = "secondary"
            btn_interactive = False
        else:
            btn_label = "stop training"
            btn_variant = "stop"
            btn_interactive = True

        return {
            "progress": progress,
            "status": state.status,
            "eta": state.calculate_eta(),
            "step": state.step,
            "max_step": state.max_step,
            "epoch": state.epoch,
            "max_epoch": state.max_epoch,
            "running": running,
            "stopping": stopping,
            "btn_label": btn_label,
            "btn_variant": btn_variant,
            "btn_interactive": btn_interactive,
        }
