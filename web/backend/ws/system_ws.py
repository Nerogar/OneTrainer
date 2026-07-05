import asyncio
import logging

from web.backend.ws.connection_manager import ConnectionManager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

METRICS_INTERVAL_S = 1.0

manager = ConnectionManager(name="System metrics WebSocket")

router = APIRouter()


@router.websocket("/ws/system")
async def system_ws(websocket: WebSocket) -> None:
    await manager.connect(websocket)

    from web.backend.services.monitor_service import MonitorService

    monitor = MonitorService.get_instance()

    try:
        while True:
            metrics = await asyncio.to_thread(monitor.get_metrics)
            await websocket.send_json({"type": "metrics", "data": metrics})
            await asyncio.sleep(METRICS_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.debug("System metrics WebSocket connection closed unexpectedly")
    finally:
        await manager.disconnect(websocket)
