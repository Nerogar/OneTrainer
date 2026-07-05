import asyncio
import logging
import os
import signal
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "version": request.app.version}


@router.post("/shutdown")
async def shutdown(request: Request):
    expected = os.environ.get("OT_SHUTDOWN_TOKEN", "")
    provided = request.headers.get("x-shutdown-token", "")
    if not expected or provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    _cleanup_before_exit()
    asyncio.get_running_loop().call_later(0.5, _exit)
    return {"status": "shutting_down"}


def _cleanup_before_exit() -> None:
    with suppress(Exception):
        from web.backend.services.trainer_service import TrainerService

        trainer_service = TrainerService.get_instance()
        status = trainer_service.get_status()
        if status["status"] in ("running", "stopping"):
            logger.info("Shutdown requested -- stopping active training run")
            trainer_service.stop_training()

        trainer_service.stop_always_on_tensorboard()


def _exit() -> None:
    # Triggers uvicorn's SIGINT handler → ASGI lifespan shutdown → clean exit.
    os.kill(os.getpid(), signal.SIGINT)
