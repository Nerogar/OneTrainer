import contextlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
    logger.warning("tensorboard package not found. TensorBoard tab will return empty data. Install with: pip install tensorboard")


MAX_CACHED_ACCUMULATORS = 10


class TensorboardService(SingletonMixin):
    def __init__(self) -> None:
        self._accumulators: dict[str, EventAccumulator] = {}
        self._access_times: dict[str, float] = {}
        self._accumulator_lock = threading.Lock()
        self._tb_process: subprocess.Popen | None = None
        self._tb_port: int | None = None
        self._tb_lock = threading.Lock()

    @staticmethod
    def _resolve_log_dir(log_dir: str | None = None) -> str:
        if log_dir:
            return log_dir

        from web.backend.services.config_service import ConfigService

        config_service = ConfigService.get_instance()
        return config_service.config.workspace_dir or "workspace"

    @staticmethod
    def _is_tfevents_dir(directory: str) -> bool:
        try:
            for entry in os.scandir(directory):
                if entry.is_file() and entry.name.startswith("events.out.tfevents"):
                    return True
        except OSError:
            pass
        return False

    def list_runs(self, log_dir: str | None = None) -> list[str]:
        resolved = self._resolve_log_dir(log_dir)
        if not os.path.isdir(resolved):
            return []

        base = Path(resolved)
        runs: list[tuple[float, str]] = []
        for dirpath, _dirnames, filenames in os.walk(resolved):
            event_files = [f for f in filenames if f.startswith("events.out.tfevents")]
            if not event_files:
                continue
            try:
                newest_mtime = max(os.path.getmtime(os.path.join(dirpath, f)) for f in event_files)
            except OSError:
                continue
            rel = Path(dirpath).relative_to(base).as_posix()
            runs.append((newest_mtime, rel))

        runs.sort(key=lambda item: (-item[0], item[1]))
        return [rel for _, rel in runs]

    def _maybe_evict(self) -> None:
        while len(self._accumulators) > MAX_CACHED_ACCUMULATORS:
            oldest_key = min(self._access_times, key=self._access_times.get)  # type: ignore[arg-type]
            del self._accumulators[oldest_key]
            del self._access_times[oldest_key]
            logger.debug("Evicted accumulator cache for: %s", oldest_key)

    def _get_accumulator(self, run_dir: str) -> "EventAccumulator | None":
        if not _HAS_TENSORBOARD:
            return None

        with self._accumulator_lock:
            if run_dir not in self._accumulators:
                self._maybe_evict()
                acc = EventAccumulator(run_dir)
                acc.Reload()
                self._accumulators[run_dir] = acc
            else:
                self._accumulators[run_dir].Reload()

            self._access_times[run_dir] = time.time()
            return self._accumulators[run_dir]

    def _run_path(self, run_name: str, log_dir: str | None = None) -> str:
        resolved = self._resolve_log_dir(log_dir)
        return os.path.join(resolved, run_name)

    def list_tags(self, run_name: str, log_dir: str | None = None) -> list[str]:
        run_path = self._run_path(run_name, log_dir)
        if not os.path.isdir(run_path):
            return []

        acc = self._get_accumulator(run_path)
        if acc is None:
            return []

        tags = acc.Tags()
        return sorted(tags.get("scalars", []))

    def get_scalars(
        self,
        run_name: str,
        tag: str,
        after_step: int = 0,
        log_dir: str | None = None,
    ) -> list[dict]:
        run_path = self._run_path(run_name, log_dir)
        if not os.path.isdir(run_path):
            return []

        acc = self._get_accumulator(run_path)
        if acc is None:
            return []

        try:
            events = acc.Scalars(tag)
        except KeyError:
            return []

        return [
            {
                "wall_time": event.wall_time,
                "step": event.step,
                "value": event.value,
            }
            for event in events
            if event.step > after_step
        ]

    # ---- Process management ----

    def _is_process_alive(self) -> bool:
        return self._tb_process is not None and self._tb_process.poll() is None

    @staticmethod
    def _wait_until_responsive(url: str, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:  # noqa: BLE001, PERF203
                time.sleep(0.3)
        return False

    @staticmethod
    def _find_tensorboard_executable() -> str | None:
        """Locate the tensorboard CLI - prefer the one in our active venv."""
        # 1. Same directory as the running Python interpreter (venv)
        venv_dir = Path(sys.executable).parent
        for candidate in ("tensorboard.exe", "tensorboard"):
            p = venv_dir / candidate
            if p.is_file():
                return str(p)
        # 2. PATH lookup
        return shutil.which("tensorboard")

    def ensure_running(self) -> dict:
        """Start tensorboard if not already running. Returns {ok, url, port, error}."""
        from web.backend.services.config_service import ConfigService

        config_service = ConfigService.get_instance()
        config = config_service.config
        log_dir = config.workspace_dir or "workspace"
        port = int(getattr(config, "tensorboard_port", 6006) or 6006)
        expose = bool(getattr(config, "tensorboard_expose", False))

        with self._tb_lock:
            if self._is_process_alive() and self._tb_port == port:
                url = f"http://localhost:{port}/"
                return {"ok": True, "url": url, "port": port, "already_running": True}

            tb_exe = self._find_tensorboard_executable()
            if tb_exe is None:
                return {
                    "ok": False,
                    "error": "tensorboard CLI not found. Install with: pip install tensorboard",
                }

            host = "0.0.0.0" if expose else "localhost"
            args = [
                tb_exe,
                "--logdir",
                log_dir,
                "--port",
                str(port),
                "--host",
                host,
            ]
            try:
                self._tb_process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._tb_port = port
            except (OSError, FileNotFoundError) as exc:
                return {"ok": False, "error": f"Failed to launch tensorboard: {exc}"}

            url = f"http://localhost:{port}/"
            if not self._wait_until_responsive(url, timeout=15.0):
                return {
                    "ok": False,
                    "error": f"tensorboard did not respond at {url} within 15s",
                    "url": url,
                    "port": port,
                }
            return {"ok": True, "url": url, "port": port, "already_running": False}

    def stop(self) -> dict:
        with self._tb_lock:
            if not self._is_process_alive():
                self._tb_process = None
                self._tb_port = None
                return {"ok": True, "was_running": False}
            try:
                self._tb_process.terminate()
                self._tb_process.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    self._tb_process.kill()
            self._tb_process = None
            self._tb_port = None
            return {"ok": True, "was_running": True}

    def clear_cache(self, run_name: str | None = None, log_dir: str | None = None) -> None:
        with self._accumulator_lock:
            if run_name:
                run_path = self._run_path(run_name, log_dir)
                self._accumulators.pop(run_path, None)
                self._access_times.pop(run_path, None)
            else:
                self._accumulators.clear()
                self._access_times.clear()
