import json
import os

from web.backend.paths import SECRETS_PATH
from web.backend.services.config_service import ConfigService

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/secrets", tags=["secrets"])

_SENSITIVE_FIELDS = {"huggingface_token", "api_key", "password"}

_MASKED_PREFIX = "<MASKED>"


class CloudSecretsBody(BaseModel):
    api_key: str | None = ""
    host: str | None = ""
    port: str | None = ""
    user: str | None = "root"
    id: str | None = ""
    key_file: str | None = ""
    password: str | None = ""


class SecretsBody(BaseModel):
    huggingface_token: str = ""
    cloud: CloudSecretsBody | None = None


def _mask_value(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return f"{_MASKED_PREFIX}****"
    return f"{_MASKED_PREFIX}{'*' * (len(value) - 4)}{value[-4:]}"


def _mask_secrets(data: dict) -> dict:
    masked = {}
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, dict):
            masked[key] = _mask_secrets(value)
        elif isinstance(value, str) and key in _SENSITIVE_FIELDS:
            masked[key] = _mask_value(value)
        else:
            masked[key] = value
    return masked


@router.get("")
def get_secrets() -> dict:
    if not os.path.isfile(SECRETS_PATH):
        service = ConfigService.get_instance()
        raw = service.config.secrets.to_dict()
        return _mask_secrets(raw)

    try:
        with open(SECRETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read secrets: {exc}") from exc

    return _mask_secrets(data)


@router.put("")
def save_secrets(body: SecretsBody) -> dict:
    data = body.model_dump()

    existing: dict = {}
    if os.path.isfile(SECRETS_PATH):
        try:
            with open(SECRETS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged = _merge_secrets(data, existing)

    service = ConfigService.get_instance()
    service.config.secrets.from_dict(merged)

    try:
        os.makedirs(os.path.dirname(SECRETS_PATH), exist_ok=True)
        with open(SECRETS_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=4)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write secrets: {exc}") from exc

    return _mask_secrets(merged)


def _merge_secrets(incoming: dict, existing: dict) -> dict:
    merged = {}
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            merged[key] = _merge_secrets(value, existing[key])
        elif isinstance(value, str) and key in _SENSITIVE_FIELDS and _is_masked(value):
            merged[key] = existing.get(key, "")
        else:
            merged[key] = value
    for key, value in existing.items():
        if key not in merged:
            merged[key] = value
    return merged


def _is_masked(value: str) -> bool:
    if not value:
        return False
    return value.startswith(_MASKED_PREFIX)
