import logging

from web.backend.ws.connection_manager import BroadcastBridge, ConnectionManager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

manager = ConnectionManager(name="Training WebSocket")
bridge = BroadcastBridge(manager, name="training")
broadcast_sync = bridge.broadcast_sync

router = APIRouter()


@router.websocket("/ws/training")
async def training_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    bridge.capture_event_loop()

    from web.backend.services.trainer_service import TrainerService

    TrainerService.get_instance().set_ws_broadcast(broadcast_sync)

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug("Received client message on training WS: %s", data)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
