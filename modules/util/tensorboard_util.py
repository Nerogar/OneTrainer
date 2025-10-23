import contextlib
import os
import subprocess
import sys
import threading
from pathlib import Path

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TensorboardMode import TensorboardMode


class TensorboardManager:
    """Singleton manager for Tensorboard processes across GUI, CLI, and Cloud training"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._process = None
        self._config = None
        self._mode = None
        self._previous_mode = None
        self._initialized = True

    def should_start(self, config: TrainConfig, is_training: bool = False) -> bool:
        """Determine if tensorboard should start based on mode and context"""
        # Skip if cloud tunnel is enabled
        if config.cloud.enabled and config.cloud.tensorboard_tunnel:
            return False

        mode = config.tensorboard_mode

        if mode == TensorboardMode.OFF:
            return False
        elif mode == TensorboardMode.ALWAYS_ON:
            return not is_training  # Start only if not already running during training
        elif mode == TensorboardMode.TRAIN_ONLY:
            return is_training

        return False

    def start(self, config: TrainConfig, mode: TensorboardMode | None = None) -> bool:
        """Start tensorboard if not already running"""
        if self._process is not None:
            return False  # Already running

        # Use provided mode or config's mode
        effective_mode = mode or config.tensorboard_mode

        # Check if we should start based on mode
        if effective_mode == TensorboardMode.OFF:
            return False

        # Skip if cloud tunnel is enabled
        if config.cloud.enabled and config.cloud.tensorboard_tunnel:
            return False

        self._process = start_filtered_tensorboard(config)
        self._config = config
        self._mode = effective_mode
        self._previous_mode = effective_mode
        return self._process is not None

    def stop(self) -> bool:
        """Stop tensorboard if running"""
        if self._process is None:
            return False

        stop_tensorboard(self._process)
        self._process = None
        self._config = None
        self._mode = None
        return True

    def restart(self, config: TrainConfig) -> bool:
        """Restart tensorboard with new config"""
        self.stop()
        return self.start(config)

    def is_running(self) -> bool:
        """Check if tensorboard is currently running"""
        return self._process is not None

    def get_mode(self) -> TensorboardMode | None:
        """Get current tensorboard mode"""
        return self._mode

    def handle_training_start(self, config: TrainConfig):
        """Handle tensorboard when training starts"""
        # If TRAIN_ONLY mode, start it
        if config.tensorboard_mode == TensorboardMode.TRAIN_ONLY:
            if not (config.cloud.enabled and config.cloud.tensorboard_tunnel):
                self.start(config)

    def handle_training_end(self, config: TrainConfig):
        """Handle tensorboard when training ends"""
        # If TRAIN_ONLY mode, stop it
        if config.tensorboard_mode == TensorboardMode.TRAIN_ONLY:
            self.stop()

        # If ALWAYS_ON mode, restart it (in case it was stopped)
        elif config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
            if not (config.cloud.enabled and config.cloud.tensorboard_tunnel):
                if not self.is_running():
                    self.start(config)

    def handle_mode_change(self, config: TrainConfig, is_training: bool = False):
        """Handle tensorboard mode changes"""
        new_mode = config.tensorboard_mode

        # Initialize previous mode on first call
        if self._previous_mode is None:
            self._previous_mode = new_mode
            # Start if ALWAYS_ON and not training
            if new_mode == TensorboardMode.ALWAYS_ON and not is_training:
                if not (config.cloud.enabled and config.cloud.tensorboard_tunnel):
                    self.start(config)
            return

        old_mode = self._previous_mode

        if new_mode == old_mode:
            return

        # Skip if cloud tunnel
        if config.cloud.enabled and config.cloud.tensorboard_tunnel:
            self.stop()
            self._previous_mode = new_mode
            return

        # During training
        if is_training:
            if new_mode == TensorboardMode.OFF:
                self.stop()
            elif old_mode == TensorboardMode.OFF and new_mode in (TensorboardMode.TRAIN_ONLY, TensorboardMode.ALWAYS_ON):
                self.start(config)
            self._previous_mode = new_mode
            return

        # Not training
        if old_mode == TensorboardMode.ALWAYS_ON:
            self.stop()

        if new_mode == TensorboardMode.ALWAYS_ON:
            self.start(config)

        self._previous_mode = new_mode

    def handle_workspace_change(self, config: TrainConfig):
        """Handle workspace directory changes"""
        if config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
            if not (config.cloud.enabled and config.cloud.tensorboard_tunnel):
                print("Restarting Tensorboard due to workspace change")
                self.restart(config)


def get_tensorboard_args(config: TrainConfig) -> list[str]:
    """
    Generates args required to start Tensorboard
    """
    tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
    tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")

    os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)

    args = [
        tensorboard_executable,
        "--logdir",
        tensorboard_log_dir,
        "--port",
        str(config.tensorboard_port),
        "--samples_per_plugin=images=100,scalars=10000",
    ]

    if config.tensorboard_expose:
        args.append("--bind_all")

    return args


def start_filtered_tensorboard(config: TrainConfig) -> subprocess.Popen | None:
    """Start Tensorboard filtering the annoying warnings Google doesnt plan to fix"""
    try:
        print("Starting Tensorboard please wait...")
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning:tensorboard.default'

        process = subprocess.Popen(
            get_tensorboard_args(config),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        def filter_output():
            for line in iter(process.stdout.readline, ''):
                # https://github.com/tensorflow/tensorboard/issues/7003
                line_lower = line.lower().strip()
                if (
                    "pkg_resources" in line_lower or
                    "installation not found" in line_lower
                ):
                    continue  # Skip these lines

                if line.strip():
                    print(line, end='')

        threading.Thread(target=filter_output, daemon=True).start()

        return process

    except Exception as e:
        print(f"Failed to start Tensorboard: {e}")
        return None


def stop_tensorboard(process: subprocess.Popen | None):
    """Gracefully stop the Tensorboard subprocess"""
    if not process:
        return

    try:
        print("Stopping Tensorboard...")
        process.terminate()

        try:
            process.wait(timeout=5)
            print("Tensorboard stopped.")
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
            try:
                process.wait(timeout=5)
                print("Tensorboard killed.")
            except subprocess.TimeoutExpired:
                print("Failed to kill Tensorboard within timeout.")

    except Exception as e:
        print(f"Error stopping Tensorboard: {e}")
    finally:
        with contextlib.suppress(Exception):
            if process.stdout:
                process.stdout.close()
