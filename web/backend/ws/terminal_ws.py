import asyncio
import logging
from contextlib import suppress

from web.backend.ws.connection_manager import BroadcastBridge, ConnectionManager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

manager = ConnectionManager(name="Terminal WebSocket")
bridge = BroadcastBridge(manager, name="terminal")
broadcast_sync = bridge.broadcast_sync

router = APIRouter()


@router.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    bridge.capture_event_loop()

    from web.backend.services.log_service import LogService

    svc = LogService.get_instance()
    svc.set_ws_broadcast(broadcast_sync)
    svc.set_event_loop(asyncio.get_running_loop())

    history = svc.get_history()
    for entry in history:
        with suppress(Exception):
            await websocket.send_json({"type": "log", "data": entry})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
