import asyncio
import logging
from contextlib import suppress

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, name: str = "WebSocket") -> None:
        self._connections: list[WebSocket] = []
        self._lock: asyncio.Lock | None = None
        self._name = name

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._get_lock():
            self._connections.append(websocket)
        logger.info("%s client connected (%s total)", self._name, len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._get_lock():
            with suppress(ValueError):
                self._connections.remove(websocket)
        logger.info("%s client disconnected (%s remaining)", self._name, len(self._connections))

    async def broadcast(self, message: dict) -> None:
        async with self._get_lock():
            stale: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:  # noqa: BLE001, PERF203
                    stale.append(ws)

            for ws in stale:
                with suppress(ValueError):
                    self._connections.remove(ws)

            if stale:
                logger.debug("Removed %d stale %s connection(s)", len(stale), self._name)

    @property
    def active_count(self) -> int:
        return len(self._connections)


class BroadcastBridge:
    def __init__(self, manager: ConnectionManager, name: str = "broadcast") -> None:
        self._manager = manager
        self._name = name
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def capture_event_loop(self) -> None:
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No running event loop found when capturing for %s", self._name)

    def broadcast_sync(self, message: dict) -> None:
        if self._manager.active_count == 0:
            return

        loop = self._event_loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._manager.broadcast(message), loop)
            future.add_done_callback(self._done_callback)
        else:
            logger.debug("No active event loop — dropping %s message", self._name)

    def _done_callback(self, future: asyncio.Future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.error("Error in %s broadcast: %s", self._name, exc, exc_info=exc)
