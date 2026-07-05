import contextlib
import json
import os
import random
import threading
import time
import uuid
from collections import OrderedDict

from web.backend.paths import CONCEPTS_DIR
from web.backend.services.concept_service import ConceptService
from web.backend.services.config_service import ConfigService
from web.backend.utils.path_security import validate_path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/concepts", tags=["concepts"])

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

_THUMBNAIL_CACHE_MAXSIZE = 256
_THUMBNAIL_CACHE_TTL = 60
_thumbnail_cache: OrderedDict[str, tuple[float, str | None]] = OrderedDict()
_thumbnail_cache_lock = threading.Lock()


def _is_valid_image(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in _IMAGE_EXTENSIONS and not name.endswith("-masklabel.png") and not name.endswith("-condlabel.png")


def _pick_thumbnail(dir_path: str, include_subdirectories: bool = False) -> str | None:
    cache_key = f"{dir_path}|{include_subdirectories}"
    now = time.monotonic()

    with _thumbnail_cache_lock:
        if cache_key in _thumbnail_cache:
            ts, value = _thumbnail_cache[cache_key]
            if now - ts < _THUMBNAIL_CACHE_TTL:
                _thumbnail_cache.move_to_end(cache_key)
                return value
            else:
                del _thumbnail_cache[cache_key]

    if not os.path.isdir(dir_path):
        result = None
    else:
        candidates: list[str] = []
        try:
            if include_subdirectories:
                for root, dirs, files in os.walk(dir_path):
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    candidates.extend(os.path.join(root, fname) for fname in files if _is_valid_image(fname))
            else:
                candidates.extend(
                    entry.path for entry in os.scandir(dir_path) if entry.is_file() and _is_valid_image(entry.name)
                )
        except PermissionError:
            result = None
        else:
            result = random.choice(candidates) if candidates else None

    with _thumbnail_cache_lock:
        _thumbnail_cache[cache_key] = (now, result)
        _thumbnail_cache.move_to_end(cache_key)
        while len(_thumbnail_cache) > _THUMBNAIL_CACHE_MAXSIZE:
            _thumbnail_cache.popitem(last=False)

    return result


def invalidate_thumbnail_cache() -> None:
    with _thumbnail_cache_lock:
        _thumbnail_cache.clear()


@router.get("/thumbnail")
def get_thumbnail(
    path: str = Query(..., description="Directory path to scan for images"),
    include_subdirectories: bool = Query(False, description="Include images from subdirectories"),
):
    path = validate_path(path, allow_file=False)
    chosen = _pick_thumbnail(path, include_subdirectories)
    if chosen is None:
        raise HTTPException(status_code=404, detail="No images found in directory")

    return FileResponse(chosen, media_type="image/*")


@router.get("/images")
def list_images(
    path: str = Query(..., description="Directory path to scan for images"),
    offset: int = Query(0, ge=0, description="Start index"),
    limit: int = Query(50, ge=1, le=10000, description="Max images to return"),
    include_subdirectories: bool = Query(False, description="Include images from subdirectories"),
):
    path = validate_path(path, allow_file=False)

    entries: list[dict] = []
    try:
        if include_subdirectories:
            for root, dirs, files in os.walk(path):
                # Skip hidden directories
                dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
                for fname in sorted(files):
                    if _is_valid_image(fname):
                        full_path = os.path.join(root, fname)
                        stem = os.path.splitext(full_path)[0]
                        caption_path = stem + ".txt"
                        caption: str | None = None
                        if os.path.isfile(caption_path):
                            try:
                                with open(caption_path, "r", encoding="utf-8") as fh:
                                    caption = fh.read().strip()
                            except Exception:
                                caption = None
                        entries.append({
                            "filename": fname,
                            "path": full_path.replace("\\", "/"),
                            "caption": caption,
                        })
        else:
            for entry in sorted(os.scandir(path), key=lambda e: e.name):
                if entry.is_file() and _is_valid_image(entry.name):
                    stem = os.path.splitext(entry.path)[0]
                    caption_path = stem + ".txt"
                    caption: str | None = None
                    if os.path.isfile(caption_path):
                        try:
                            with open(caption_path, "r", encoding="utf-8") as fh:
                                caption = fh.read().strip()
                        except Exception:
                            caption = None
                    entries.append({
                        "filename": entry.name,
                        "path": entry.path.replace("\\", "/"),
                        "caption": caption,
                    })
    except PermissionError as err:
        raise HTTPException(status_code=403, detail="Permission denied") from err

    total = len(entries)
    page = entries[offset:offset + limit]

    return JSONResponse({"total": total, "offset": offset, "images": page})


@router.get("/image")
def get_image(path: str = Query(..., description="Full path to an image file")):
    path = validate_path(path, allow_dir=False)

    ext = os.path.splitext(path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a supported image file")

    return FileResponse(path, media_type="image/*")


@router.get("/text-file")
def get_text_file(path: str = Query(..., description="Path to a text file")):
    path = validate_path(path, allow_dir=False)

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".txt", ".caption", ".csv"}:
        raise HTTPException(status_code=400, detail="Not a supported text file")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        return JSONResponse({"content": content})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid file encoding") from exc


@router.get("/configs")
def list_concept_configs() -> list[dict]:
    configs: list[dict] = []
    if os.path.isdir(CONCEPTS_DIR):
        for entry in sorted(os.listdir(CONCEPTS_DIR)):
            if entry.endswith(".json") and os.path.isfile(os.path.join(CONCEPTS_DIR, entry)):
                name = os.path.splitext(entry)[0]
                rel_path = f"training_concepts/{entry}"
                configs.append({"name": name, "path": rel_path})
    return configs


class CreateConfigRequest(BaseModel):
    name: str = Field(max_length=200)


@router.post("/configs")
def create_concept_config(req: CreateConfigRequest) -> dict:
    safe_name = "".join(c for c in req.name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid config name")

    os.makedirs(CONCEPTS_DIR, exist_ok=True)
    file_path = os.path.join(CONCEPTS_DIR, f"{safe_name}.json")
    rel_path = f"training_concepts/{safe_name}.json"

    if os.path.exists(file_path):
        raise HTTPException(status_code=409, detail=f"Config '{safe_name}' already exists")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    return {"name": safe_name, "path": rel_path}


def _resolve_concept_path(path: str | None) -> str:
    if path:
        # Concept files live under the project; restrict to JSON files via the
        # standard path validator to keep the override flow sandboxed.
        return validate_path(path, allow_file=True)
    service = ConfigService.get_instance()
    concept_path = service.config.concept_file_name
    if not concept_path:
        raise HTTPException(status_code=422, detail="No concept_file_name configured")
    return concept_path


@router.get("")
def get_concepts(
    path: str | None = Query(
        None,
        description="Override the concept file path; defaults to the global config value.",
    ),
) -> list[dict]:
    concept_path = _resolve_concept_path(path)
    concept_service = ConceptService()
    try:
        return concept_service.load_concepts(concept_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Concept file not found: {concept_path}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("")
def save_concepts(
    concepts: list[dict],
    path: str | None = Query(
        None,
        description="Override the concept file path; defaults to the global config value.",
    ),
) -> dict:
    concept_path = _resolve_concept_path(path)
    concept_service = ConceptService()
    try:
        concept_service.save_concepts(concept_path, concepts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    invalidate_thumbnail_cache()

    return {"saved": len(concepts), "path": concept_path}


_active_scans: dict[str, threading.Event] = {}
_active_scans_lock = threading.Lock()


class StatsRequest(BaseModel):
    path: str
    include_subdirectories: bool = False
    advanced: bool = False


@router.post("/stats")
def scan_concept_stats(req: StatsRequest):
    req.path = validate_path(req.path, allow_file=False)

    try:
        from modules.util import concept_stats
        from modules.util.config.ConceptConfig import ConceptConfig
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Backend modules not available: {exc}",
        ) from exc

    scan_id = uuid.uuid4().hex
    cancel_flag = threading.Event()
    with _active_scans_lock:
        _active_scans[scan_id] = cancel_flag

    try:
        start_time = time.perf_counter()
        max_scan_seconds = 120

        concept_config = ConceptConfig.default_values()
        concept_config.path = req.path
        concept_config.include_subdirectories = req.include_subdirectories

        stats_dict = concept_stats.init_concept_stats(req.advanced)

        subfolders = [req.path]

        for folder in subfolders:
            if cancel_flag.is_set():
                break
            if time.perf_counter() - start_time > max_scan_seconds:
                stats_dict["force_cancelled"] = True
                break
            stats_dict = concept_stats.folder_scan(
                folder, stats_dict, req.advanced, concept_config,
                start_time, max_scan_seconds, cancel_flag,
            )
            if req.include_subdirectories and not cancel_flag.is_set():
                with contextlib.suppress(PermissionError):
                    subfolders.extend(
                        entry.path for entry in os.scandir(folder) if entry.is_dir()
                    )

        stats_dict["processing_time"] = round(time.perf_counter() - start_time, 3)
        stats_dict["scan_id"] = scan_id

        return JSONResponse(stats_dict)
    finally:
        with _active_scans_lock:
            _active_scans.pop(scan_id, None)


@router.delete("/stats/cancel")
def cancel_concept_stats(scan_id: str = Query("", description="Scan ID to cancel; empty cancels all active scans")):
    with _active_scans_lock:
        if scan_id:
            flag = _active_scans.get(scan_id)
            if flag is None:
                raise HTTPException(status_code=404, detail=f"No active scan with id: {scan_id}")
            flag.set()
            return {"cancelled": True, "scan_id": scan_id}
        else:
            cancelled_ids = list(_active_scans.keys())
            for flag in _active_scans.values():
                flag.set()
            return {"cancelled": True, "scan_ids": cancelled_ids}


# ---- Caption save ----


class SaveCaptionRequest(BaseModel):
    image_path: str
    caption: str


@router.post("/save-caption")
def save_caption(req: SaveCaptionRequest):
    """Write a .txt caption file next to the given image."""
    safe = validate_path(req.image_path, must_exist=True, allow_dir=False)
    base, _ = os.path.splitext(safe)
    caption_path = base + ".txt"
    try:
        with open(caption_path, "w", encoding="utf-8") as fh:
            fh.write(req.caption)
        return {"ok": True, "saved": caption_path}
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write caption: {exc}") from exc


# ---- Augmentation preview ----


class AugPreviewRequest(BaseModel):
    image_path: str
    image: dict  # ConceptImageConfig as plain dict
    seed: int = Field(default_factory=lambda: random.randint(0, 2**30))


def _apply_image_augmentations(img, image_cfg: dict, rng: random.Random):
    """Lightweight PIL approximation of the per-concept image augmentations.

    Mirrors the most common operations from mgds (RandomBrightness/Contrast/
    Saturation/Hue/Flip/Rotate) but stays self-contained for fast preview.
    """
    from PIL import ImageEnhance, ImageOps

    def _strength(enable_random: str, enable_fixed: str, max_strength: float) -> float:
        if image_cfg.get(enable_random):
            return rng.uniform(-max_strength, max_strength)
        if image_cfg.get(enable_fixed):
            return max_strength
        return 0.0

    if image_cfg.get("enable_random_flip") and rng.random() < 0.5 or image_cfg.get("enable_fixed_flip"):
        img = ImageOps.mirror(img)

    if image_cfg.get("enable_random_rotate") or image_cfg.get("enable_fixed_rotate"):
        max_angle = float(image_cfg.get("random_rotate_max_angle", 0) or 0)
        if max_angle > 0:
            angle = rng.uniform(-max_angle, max_angle) if image_cfg.get("enable_random_rotate") else max_angle
            img = img.rotate(angle, resample=2, expand=False)  # 2 = BILINEAR

    s = _strength(
        "enable_random_brightness",
        "enable_fixed_brightness",
        float(image_cfg.get("random_brightness_max_strength", 0) or 0),
    )
    if s != 0.0:
        img = ImageEnhance.Brightness(img.convert("RGB")).enhance(1.0 + s)

    s = _strength(
        "enable_random_contrast",
        "enable_fixed_contrast",
        float(image_cfg.get("random_contrast_max_strength", 0) or 0),
    )
    if s != 0.0:
        img = ImageEnhance.Contrast(img.convert("RGB")).enhance(1.0 + s)

    s = _strength(
        "enable_random_saturation",
        "enable_fixed_saturation",
        float(image_cfg.get("random_saturation_max_strength", 0) or 0),
    )
    if s != 0.0:
        img = ImageEnhance.Color(img.convert("RGB")).enhance(1.0 + s)

    s = _strength(
        "enable_random_hue",
        "enable_fixed_hue",
        float(image_cfg.get("random_hue_max_strength", 0) or 0),
    )
    if s != 0.0:
        from PIL import Image as _Image

        hsv = img.convert("HSV")
        h, sat, v = hsv.split()
        h = h.point(lambda px: int((px + s * 128) % 256))
        img = _Image.merge("HSV", (h, sat, v)).convert("RGB")

    return img


@router.post("/augmentation-preview")
def augmentation_preview(req: AugPreviewRequest):
    """Apply augmentations to a single image and return the result as base64 PNG."""
    import base64
    import io

    from PIL import Image

    safe_path = validate_path(req.image_path, must_exist=True, allow_dir=False)
    rng = random.Random(req.seed)

    try:
        with Image.open(safe_path) as src:
            src.load()
            preview = _apply_image_augmentations(src.convert("RGB"), req.image, rng)

        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"ok": True, "image_base64": encoded, "seed": req.seed}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Augmentation preview failed: {exc}") from exc
