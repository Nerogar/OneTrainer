import asyncio
import io
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

_log_reentrant = threading.local()


class _TeeWriter(io.TextIOBase):
    def __init__(self, original: io.TextIOBase, log_service: "LogService") -> None:
        self._original = original
        self._log_service = log_service

    def write(self, s: str) -> int:
        if not s:
            return 0
        result = self._original.write(s)
        self._original.flush()

        self._log_service.append(s)

        return result

    def flush(self) -> None:
        self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False


class _WebSocketLogHandler(logging.Handler):
    def __init__(self, log_service: "LogService") -> None:
        super().__init__()
        self._log_service = log_service
        self.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._log_service.append(msg + "\n")
        except Exception:
            self.handleError(record)


class LogService(SingletonMixin):
    def __init__(self) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._ws_broadcast: Callable[[dict], None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True

        sys.stdout = _TeeWriter(sys.stdout, self)  # type: ignore[assignment]
        sys.stderr = _TeeWriter(sys.stderr, self)  # type: ignore[assignment]

        handler = _WebSocketLogHandler(self)
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

    def set_ws_broadcast(self, broadcast_fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = broadcast_fn

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._event_loop = loop

    def get_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    def append(self, text: str) -> None:
        if getattr(_log_reentrant, "in_append", False):
            return
        _log_reentrant.in_append = True
        try:
            entry = {"text": text, "ts": time.time()}
            with self._lock:
                self._buffer.append(entry)

            message = {"type": "log", "data": entry}
            if self._ws_broadcast is not None:
                self._ws_broadcast(message)
        finally:
            _log_reentrant.in_append = False
