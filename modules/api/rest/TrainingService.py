import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from threading import Lock, Thread
from time import monotonic
from uuid import uuid4

from modules.api.rest.ApiError import ConflictError, NoActiveRunError
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig

ACTIVE_STATES = frozenset({"starting", "running", "stopping"})


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class _Run:
    run_id: str
    state: str
    commands: TrainCommands
    message: str = ""
    epoch: int = 0
    max_epochs: int = 0
    epoch_step: int = 0
    epoch_length: int = 0
    step: int = 0
    started_at: str = ""
    started_monotonic: float = 0.0
    ended_monotonic: float | None = None
    error: dict | None = None


class TrainingService:
    """
    Adapts OneTrainer's TrainCallbacks/TrainCommands pair to HTTP.

    Callbacks come in from the trainer thread and are accumulated into a
    pollable status dict; commands go out to the TrainCommands object the
    trainer polls at step boundaries.

    Locking rules -- the only two ways this class goes wrong:
      1. Never hold self._lock while calling into the trainer or a subclass
         hook. Trainer calls take minutes; hooks are somebody else's code.
      2. status() builds its entire dict under the lock, so a poll can never
         observe a half-updated run.
    """

    def __init__(self, trainer_factory=None) -> None:
        self._lock = Lock()
        self._run: _Run | None = None
        self._trainer_factory = trainer_factory

    # --- queries ---

    def status(self) -> dict:
        with self._lock:
            run = self._run
            if run is None:
                return {
                    "run_id": None,
                    "state": "idle",
                    "message": "",
                    "epoch": 0,
                    "max_epochs": 0,
                    "epoch_step": 0,
                    "epoch_length": 0,
                    "step": 0,
                    "max_steps": 0,
                    "started_at": None,
                    "elapsed_seconds": 0.0,
                    "error": None,
                }
            end = run.ended_monotonic if run.ended_monotonic is not None else monotonic()
            return {
                "run_id": run.run_id,
                "state": run.state,
                "message": run.message,
                "epoch": run.epoch,
                "max_epochs": run.max_epochs,
                "epoch_step": run.epoch_step,
                "epoch_length": run.epoch_length,
                "step": run.step,
                "max_steps": run.epoch_length * run.max_epochs,
                "started_at": run.started_at,
                "elapsed_seconds": round(end - run.started_monotonic, 3),
                "error": dict(run.error) if run.error is not None else None,
            }

    # --- lifecycle ---

    def start(self, config: TrainConfig) -> str:
        with self._lock:
            if self._run is not None and self._run.state in ACTIVE_STATES:
                raise ConflictError("A training run is already active", run_id=self._run.run_id)
            run = _Run(
                run_id=uuid4().hex[:16],
                state="starting",
                commands=TrainCommands(),
                started_at=_utcnow_iso(),
                started_monotonic=monotonic(),
            )
            self._run = run

        Thread(
            target=self._worker,
            args=(run, config),
            daemon=True,
            name=f"onetrainer-run-{run.run_id}",
        ).start()
        return run.run_id

    # --- commands ---
    # Each is a single call onto the TrainCommands object the trainer polls at
    # step boundaries. Nothing here happens synchronously; that is why the HTTP
    # layer answers 202 rather than 200.

    def stop(self) -> None:
        with self._lock:
            run = self._require_active_locked()
            run.state = "stopping"
            commands = run.commands
        commands.stop()

    def sample(self, sample: SampleConfig | None = None) -> None:
        if sample is None:
            self._dispatch(lambda commands: commands.sample_default())
        else:
            self._dispatch(lambda commands: commands.sample_custom(sample))

    def backup(self) -> None:
        self._dispatch(lambda commands: commands.backup())

    def save(self) -> None:
        self._dispatch(lambda commands: commands.save())

    def _dispatch(self, action: Callable[[TrainCommands], None]) -> None:
        with self._lock:
            commands = self._require_active_locked().commands
        action(commands)

    def _require_active_locked(self) -> _Run:
        # Caller must hold self._lock.
        if self._run is None or self._run.state not in ACTIVE_STATES:
            raise NoActiveRunError("There is no active training run")
        return self._run

    def _worker(self, run: _Run, config: TrainConfig) -> None:
        self._safe_hook(self._on_run_begin, run.run_id, config)
        try:
            trainer_factory = self._trainer_factory
            if trainer_factory is None:
                # Lazy: keeps diffusers/transformers/model setup out of import
                # time, so create_app() and the tests stay cheap.
                from modules.util import create

                trainer_factory = create.create_trainer

            callbacks = TrainCallbacks(
                on_update_status=partial(self._handle_status, run),
                on_update_train_progress=partial(self._handle_progress, run),
                on_sample_default=partial(self._handle_sample_default, run),
            )
            trainer = trainer_factory(config, callbacks, run.commands)

            trainer.start()
            self._set_state(run, "running", expect="starting")
            trainer.train()

            # Same condition as scripts/train.py:55 -- deliberately identical to
            # the CLI. Cancellation reaches us as a stop flag, not KeyboardInterrupt.
            canceled = run.commands.get_stop_command()
            if not canceled or config.backup_before_save:
                trainer.end()
            self._finish(run, "canceled" if canceled else "completed")
        except Exception as e:
            print(f"training run {run.run_id} failed")
            traceback.print_exc()
            self._finish(run, "failed", error={"type": type(e).__name__, "message": str(e)})
        finally:
            with self._lock:
                final_state = run.state
            self._safe_hook(self._on_run_end, run.run_id, final_state)

    # --- callback sinks ---

    def _handle_status(self, run: _Run, status: str) -> None:
        with self._lock:
            run.message = status
        self._safe_hook(self._on_status, status)

    def _handle_progress(self, run: _Run, train_progress, max_step: int, max_epoch: int) -> None:
        # max_step is the *epoch length* (GenericTrainer.py:832), not a total
        # step count. Total steps is epoch_length * max_epochs, computed in status().
        with self._lock:
            run.epoch = train_progress.epoch
            run.epoch_step = train_progress.epoch_step
            run.step = train_progress.global_step
            run.epoch_length = max_step
            run.max_epochs = max_epoch
        self._safe_hook(self._on_progress, train_progress, max_step, max_epoch)

    def _handle_sample_default(self, run: _Run, sampler_output) -> None:
        self._safe_hook(self._on_sample_default, sampler_output)

    # --- state transitions ---

    def _set_state(self, run: _Run, state: str, expect: str | None = None) -> None:
        # expect guards against clobbering a concurrent transition: a stop()
        # during a slow model load sets "stopping", and without it the worker
        # would drop the run back to "running" once trainer.start() returned.
        with self._lock:
            if expect is not None and run.state != expect:
                return
            run.state = state

    def _finish(self, run: _Run, state: str, error: dict | None = None) -> None:
        with self._lock:
            run.state = state
            run.error = error
            run.ended_monotonic = monotonic()

    @staticmethod
    def _safe_hook(hook, *args) -> None:
        # TrainCallbacks suppresses every exception a callback raises, so a
        # broken handler would vanish without a trace. Report it here instead.
        try:
            hook(*args)
        except Exception:
            print(f"error in training service hook {getattr(hook, '__name__', hook)}")
            traceback.print_exc()

    # --- subclass extension points ---
    # Deliberately no-ops. Our Web UI fork overrides these to drive its event
    # hub, metrics store, checkpoint store and gallery without forking this class.

    def _on_run_begin(self, run_id: str, config: TrainConfig) -> None:
        pass

    def _on_run_end(self, run_id: str, state: str) -> None:
        pass

    def _on_status(self, message: str) -> None:
        pass

    def _on_progress(self, train_progress, max_step: int, max_epoch: int) -> None:
        pass

    def _on_sample_default(self, sampler_output) -> None:
        pass
