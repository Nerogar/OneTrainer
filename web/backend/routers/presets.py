import os

from web.backend.paths import PRESETS_DIR
from web.backend.routers.concepts import invalidate_thumbnail_cache
from web.backend.services.config_service import ConfigService
from web.backend.utils.path_security import base_match, validate_path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/presets", tags=["presets"])

_PRESETS_DIR_REAL = os.path.realpath(PRESETS_DIR)


def _validate_preset_path(path: str) -> str:
    canonical = validate_path(path, must_exist=False)
    if not base_match(canonical, _PRESETS_DIR_REAL):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside the presets directory",
        )
    return canonical


class PresetInfo(BaseModel):
    name: str
    path: str
    is_builtin: bool


class LoadPresetRequest(BaseModel):
    path: str


class SavePresetRequest(BaseModel):
    name: str


@router.get("", response_model=list[PresetInfo])
def list_presets() -> list[PresetInfo]:
    presets: list[PresetInfo] = []

    if not os.path.isdir(PRESETS_DIR):
        return presets

    for filename in sorted(os.listdir(PRESETS_DIR)):
        if not filename.endswith(".json"):
            continue
        name = filename.removesuffix(".json")
        is_builtin = name.startswith("#")
        full_path = os.path.join(PRESETS_DIR, filename)
        presets.append(PresetInfo(name=name, path=full_path, is_builtin=is_builtin))

    return presets


@router.post("/load")
def load_preset(body: LoadPresetRequest) -> dict:
    canonical = _validate_preset_path(body.path)

    if not os.path.isfile(canonical):
        raise HTTPException(status_code=404, detail=f"Preset file not found: {body.path}")

    service = ConfigService.get_instance()
    try:
        result = service.load_preset(canonical)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load preset: {exc}") from exc
    invalidate_thumbnail_cache()
    return result


@router.post("/save")
def save_preset(body: SavePresetRequest) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Preset name must not be empty")
    if name.startswith("#"):
        raise HTTPException(status_code=403, detail="Cannot save a preset with a name starting with '#' (reserved for built-in presets)")

    path = os.path.join(PRESETS_DIR, f"{name}.json")
    canonical = _validate_preset_path(path)

    service = ConfigService.get_instance()
    try:
        service.save_preset(canonical)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save preset: {exc}") from exc

    return {"name": name, "path": canonical}


@router.delete("/{name}")
def delete_preset(name: str) -> dict:
    if name.startswith("#"):
        raise HTTPException(status_code=403, detail="Cannot delete built-in presets")

    path = os.path.join(PRESETS_DIR, f"{name}.json")
    canonical = _validate_preset_path(path)

    if not os.path.isfile(canonical):
        raise HTTPException(status_code=404, detail=f"Preset not found: {name}")

    try:
        os.remove(canonical)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete preset: {exc}") from exc

    return {"deleted": name}
