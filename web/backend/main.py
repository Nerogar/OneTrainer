import logging
import os
import sys
from contextlib import asynccontextmanager

from web.backend.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

sys.path.insert(0, PROJECT_ROOT)

from web.backend.routers import (
    concepts,
    config,
    converter,
    health,
    mask_editor,
    presets,
    samples,
    sampling,
    secrets,
    system,
    tensorboard,
    tools,
    training,
    video_tools,
    wiki,
)
from web.backend.ws import system_ws, terminal_ws, training_ws

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    from web.backend.services.log_service import LogService

    LogService.get_instance().install()
    yield


app = FastAPI(
    title="OneTrainerWeb API",
    version="0.1.0",
    lifespan=lifespan,
)

_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
if os.environ.get("OT_ELECTRON") == "1":
    _default_origins.append("null")
_cors_origins = os.environ.get("OT_CORS_ORIGINS", "").split(",") if os.environ.get("OT_CORS_ORIGINS") else _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_routers = [
    health.router,
    config.router,
    presets.router,
    concepts.router,
    samples.router,
    secrets.router,
    tensorboard.router,
    training.router,
    wiki.router,
    system.router,
    tools.router,
    converter.router,
    video_tools.router,
    sampling.router,
    mask_editor.router,
    tools.debug_router,
]
for _router in _routers:
    app.include_router(_router, prefix="/api")

app.include_router(training_ws.router)
app.include_router(system_ws.router)
app.include_router(terminal_ws.router)
