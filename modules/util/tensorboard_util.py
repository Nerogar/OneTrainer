import contextlib
import os
import subprocess
import sys
import threading
from pathlib import Path

from modules.util.config.TrainConfig import TrainConfig


def get_tensorboard_args(config: TrainConfig) -> list[str]:
    """Get Tensorboard command line arguments"""
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
    """Start Tensorboard with filtered output"""
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
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            process.kill()
    except Exception:
        pass
