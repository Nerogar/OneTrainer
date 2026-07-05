import importlib.util
import io
import json
import logging
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone

from web.backend.paths import SECRETS_PATH
from web.backend.services.tool_service import ToolService
from web.backend.utils.gpu_info import get_gpu_static_info
from web.backend.utils.path_security import validate_path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

_MASKED_PREFIX = "<MASKED>"


class CaptionRequest(BaseModel):
    model: str = "Blip"
    folder: str
    initial_caption: str = ""
    caption_prefix: str = ""
    caption_postfix: str = ""
    mode: str = "fill"
    include_subdirectories: bool = False


class ApiCaptionRequest(BaseModel):
    backend: str = "openai"
    folder: str
    prompt: str = ""
    additional_prompts: list[str] = []
    mode: str = "fill"
    include_subdirectories: bool = False
    caption_prefix: str = ""
    caption_postfix: str = ""
    api_url: str = "http://localhost:1234/v1/chat/completions"
    api_key: str = ""
    model_name: str = ""
    system_prompt: str = ""
    temperature: float = 0.6
    max_tokens: int = -1
    enable_thinking: bool = False
    batch_size: int = 1
    requests_per_minute: int = 0
    timeout: int = 120
    pass_filename: bool = False
    pass_metadata: bool = False
    pass_current_caption: bool = False


class PreviewRequest(BaseModel):
    backend: str = ""
    model: str = ""
    folder: str = ""
    image_path: str | None = None
    prompt: str = ""
    include_subdirectories: bool = False
    api_url: str = "http://localhost:1234/v1/chat/completions"
    api_key: str = ""
    model_name: str = ""
    system_prompt: str = ""
    temperature: float = 0.6
    max_tokens: int = -1
    enable_thinking: bool = False
    timeout: int = 120
    pass_filename: bool = False
    pass_metadata: bool = False
    pass_current_caption: bool = False


class PreviewResponse(BaseModel):
    ok: bool
    caption: str = ""
    prompt_used: str = ""
    image_path: str = ""
    image_base64: str = ""
    error: str | None = None


class MaskRequest(BaseModel):
    model: str = "ClipSeg"
    folder: str
    prompt: str = ""
    mode: str = "fill"
    threshold: float = 0.3
    smooth: int = 5
    expand: int = 10
    alpha: float = 1.0
    include_subdirectories: bool = False
    model_path: str | None = None


class ToolActionResponse(BaseModel):
    ok: bool
    error: str | None = None
    task_id: str | None = None


class ToolStatusResponse(BaseModel):
    status: str
    progress: int = 0
    max_progress: int = 0
    error: str | None = None
    task_id: str | None = None


class CapabilitiesResponse(BaseModel):
    ultralytics_available: bool = False
    caption_models: list[str] = []
    mask_models: list[str] = []


class CaptionKeysBody(BaseModel):
    openai_api_key: str = ""
    openai_api_url: str = ""
    gemini_api_key: str = ""


@router.post("/tools/captions/generate", response_model=ToolActionResponse)
def generate_captions(req: CaptionRequest):
    validate_path(req.folder, must_exist=True, allow_file=False)
    service = ToolService.get_instance()
    result = service.generate_captions(req)
    return ToolActionResponse(**result)


@router.post("/tools/captions/generate-api", response_model=ToolActionResponse)
def generate_captions_api(req: ApiCaptionRequest):
    validate_path(req.folder, must_exist=True, allow_file=False)
    service = ToolService.get_instance()
    result = service.generate_captions_api(req)
    return ToolActionResponse(**result)


@router.post("/tools/captions/preview", response_model=PreviewResponse)
def preview_caption(req: PreviewRequest):
    if req.image_path:
        validate_path(req.image_path, must_exist=True, allow_dir=False)
    elif req.folder:
        validate_path(req.folder, must_exist=True, allow_file=False)
    service = ToolService.get_instance()
    result = service.preview_caption(req)
    return PreviewResponse(**result)


@router.post("/tools/masks/generate", response_model=ToolActionResponse)
def generate_masks(req: MaskRequest):
    validate_path(req.folder, must_exist=True, allow_file=False)
    service = ToolService.get_instance()
    result = service.generate_masks(req)
    return ToolActionResponse(**result)


@router.get("/tools/status", response_model=ToolStatusResponse)
def get_status():
    service = ToolService.get_instance()
    status = service.get_status()
    return ToolStatusResponse(**status)


@router.post("/tools/cancel", response_model=ToolActionResponse)
def cancel_tool():
    service = ToolService.get_instance()
    result = service.cancel()
    return ToolActionResponse(**result)


@router.get("/tools/capabilities", response_model=CapabilitiesResponse)
def get_capabilities():
    return CapabilitiesResponse(
        ultralytics_available=importlib.util.find_spec("ultralytics") is not None,
        caption_models=["Blip", "Blip2", "WD14 VIT v2", "OpenAI Compatible", "Gemini API"],
        mask_models=["ClipSeg", "Rembg", "Rembg-Human", "Hex Color", "YOLO"],
    )


def _mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return f"{_MASKED_PREFIX}****"
    return f"{_MASKED_PREFIX}{'*' * (len(value) - 4)}{value[-4:]}"


def _is_masked(value: str) -> bool:
    return bool(value) and value.startswith(_MASKED_PREFIX)


def _load_secrets() -> dict:
    if not os.path.isfile(SECRETS_PATH):
        return {}
    try:
        with open(SECRETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_secrets(data: dict) -> None:
    os.makedirs(os.path.dirname(SECRETS_PATH), exist_ok=True)
    with open(SECRETS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


@router.get("/tools/captions/keys")
def get_caption_keys():
    secrets = _load_secrets()
    return {
        "openai_api_key": _mask_key(secrets.get("caption_openai_api_key", "")),
        "openai_api_url": secrets.get("caption_openai_api_url", ""),
        "gemini_api_key": _mask_key(secrets.get("caption_gemini_api_key", "")),
    }


@router.put("/tools/captions/keys")
def save_caption_keys(body: CaptionKeysBody):
    secrets = _load_secrets()

    if body.openai_api_key and not _is_masked(body.openai_api_key):
        secrets["caption_openai_api_key"] = body.openai_api_key
    if body.openai_api_url:
        secrets["caption_openai_api_url"] = body.openai_api_url
    if body.gemini_api_key and not _is_masked(body.gemini_api_key):
        secrets["caption_gemini_api_key"] = body.gemini_api_key

    _save_secrets(secrets)
    return {
        "openai_api_key": _mask_key(secrets.get("caption_openai_api_key", "")),
        "openai_api_url": secrets.get("caption_openai_api_url", ""),
        "gemini_api_key": _mask_key(secrets.get("caption_gemini_api_key", "")),
    }


def _collect_system_info() -> str:
    lines: list[str] = []

    uname = platform.uname()
    lines.append("=== System Information ===")
    lines.append(f"OS: {uname.system} {uname.release}")
    lines.append(f"Version: {uname.version}")
    lines.append(f"Machine: {uname.machine}")
    lines.append("")

    lines.append("=== Python Environment ===")
    lines.append(f"Python Version: {sys.version}")
    lines.append(f"Python Executable: {sys.executable}")
    lines.append("")

    lines.append("=== PyTorch / CUDA ===")
    try:
        import torch

        lines.append(f"PyTorch Version: {torch.__version__}")
        lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA Version: {torch.version.cuda}")
            try:
                gpus = get_gpu_static_info()
                for gpu in gpus:
                    mem_gb = round(gpu["vram_total_mb"] / 1024, 2)
                    lines.append(f"  GPU {gpu['index']}: {gpu['name']} ({mem_gb} GB)")
            except Exception as exc:
                lines.append(f"Error querying GPUs: {exc}")
    except ImportError:
        lines.append("PyTorch not installed")
    except Exception as exc:
        lines.append(f"Error querying PyTorch: {exc}")
    lines.append("")

    lines.append("=== Memory ===")
    try:
        import psutil

        vm = psutil.virtual_memory()
        lines.append(f"Total RAM: {round(vm.total / (1024**3), 2)} GB")
        lines.append(f"Available RAM: {round(vm.available / (1024**3), 2)} GB")
    except ImportError:
        lines.append("psutil not installed -- cannot read memory info")
    except Exception as exc:
        lines.append(f"Error querying memory: {exc}")
    lines.append("")

    lines.append("=== CPU ===")
    lines.append(f"Processor: {platform.processor() or 'Unavailable'}")
    try:
        import psutil as _ps

        lines.append(f"Physical Cores: {_ps.cpu_count(logical=False)}")
        lines.append(f"Logical Cores: {_ps.cpu_count(logical=True)}")
    except ImportError:
        pass
    except Exception as exc:
        lines.append(f"Error querying CPU: {exc}")
    lines.append("")

    return "\n".join(lines)


def _collect_log_output() -> str:
    try:
        from web.backend.services.log_service import LogService

        history = LogService.get_instance().get_history()
        return "\n".join(entry["text"] for entry in history)
    except Exception as exc:
        return f"Error collecting log output: {exc}"


debug_router = APIRouter(tags=["tools"])


@debug_router.post("/tools/debug-package")
def generate_debug_package():
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            from web.backend.services.config_service import ConfigService

            config_dict = ConfigService.get_instance().export_config()
            config_json = json.dumps(config_dict, indent=2, default=str)
            zf.writestr("config.json", config_json)
        except Exception as exc:
            logger.warning("Could not include config in debug package: %s", exc)
            zf.writestr("config.json", json.dumps({"error": str(exc)}))

        try:
            system_info = _collect_system_info()
            zf.writestr("system_info.txt", system_info)
        except Exception as exc:
            logger.warning("Could not collect system info: %s", exc)
            zf.writestr("system_info.txt", f"Error: {exc}")

        try:
            log_output = _collect_log_output()
            zf.writestr("log_output.txt", log_output)
        except Exception as exc:
            logger.warning("Could not collect log output: %s", exc)
            zf.writestr("log_output.txt", f"Error: {exc}")

    buf.seek(0)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"OneTrainer_debug_{timestamp}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
