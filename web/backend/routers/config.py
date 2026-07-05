import json
import os

from web.backend.routers.concepts import invalidate_thumbnail_cache
from web.backend.services.config_service import ConfigService

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_OPTIMIZER_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "generated", "optimizer_defaults.json")
_optimizer_defaults_cache: dict | None = None


def _load_optimizer_defaults() -> dict:
    global _optimizer_defaults_cache
    if _optimizer_defaults_cache is None:
        with open(_OPTIMIZER_DEFAULTS_PATH, encoding="utf-8") as f:
            _optimizer_defaults_cache = json.load(f)
    return _optimizer_defaults_cache


_OPTIMIZER_KEY_DETAILS_PATH = os.path.join(os.path.dirname(__file__), "..", "generated", "optimizer_key_details.json")
_optimizer_key_details_cache: dict | None = None


def _load_key_detail_map() -> dict:
    global _optimizer_key_details_cache
    if _optimizer_key_details_cache is None:
        with open(_OPTIMIZER_KEY_DETAILS_PATH, encoding="utf-8") as f:
            _optimizer_key_details_cache = json.load(f)
    return _optimizer_key_details_cache


router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
def get_config() -> dict:
    service = ConfigService.get_instance()
    return service.get_config_dict()


@router.put("")
async def update_config(request: Request) -> dict:
    service = ConfigService.get_instance()
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    if "concept_file_name" in data:
        old_value = getattr(service.config, "concept_file_name", None)
        if data["concept_file_name"] != old_value:
            invalidate_thumbnail_cache()

    try:
        return service.update_config(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/validate")
async def validate_config(request: Request) -> dict:
    service = ConfigService.get_instance()
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    return service.validate_config(data)


@router.get("/defaults")
def get_defaults() -> dict:
    service = ConfigService.get_instance()
    return service.get_defaults()


@router.get("/schema")
def get_schema() -> dict:
    service = ConfigService.get_instance()
    config = service.config

    fields: dict[str, dict] = {}
    for name, var_type in config.types.items():
        type_name = getattr(var_type, "__name__", str(var_type))
        fields[name] = {
            "type": type_name,
            "nullable": config.nullables.get(name, False),
        }

    return {"fields": fields}


class ChangeOptimizerRequest(BaseModel):
    optimizer: str


@router.post("/change-optimizer")
def change_optimizer_endpoint(body: ChangeOptimizerRequest) -> dict:
    service = ConfigService.get_instance()
    try:
        return service.change_optimizer(body.optimizer)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown optimizer: {body.optimizer}") from exc


@router.get("/optimizer-params")
def get_optimizer_params() -> dict:
    all_defaults = _load_optimizer_defaults()
    key_detail_map = _load_key_detail_map()

    optimizers: dict[str, dict] = {}
    for opt_name, defaults in all_defaults.items():
        clean_defaults = {}
        for k, v in defaults.items():
            if isinstance(v, float) and v == float("inf"):
                clean_defaults[k] = "Infinity"
            else:
                clean_defaults[k] = v
        optimizers[opt_name] = {
            "keys": list(defaults.keys()),
            "defaults": clean_defaults,
        }

    return {
        "optimizers": optimizers,
        "detail_map": key_detail_map,
    }


@router.post("/export")
def export_config() -> dict:
    service = ConfigService.get_instance()
    try:
        return service.export_config()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Referenced file not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
