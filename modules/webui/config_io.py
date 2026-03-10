"""
Configuration I/O helpers for the OneTrainer Gradio WebUI.

Handles listing, loading, and saving training preset JSON files,
concept file I/O, and dataset image preview utilities.
"""

import base64
import io
import json
import os
import pathlib
from pathlib import Path

from PIL import Image as PILImage

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig

PRESET_DIR = Path("training_presets")
CONCEPT_DIR = Path("training_concepts")

SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'}


# ── Preset helpers ──────────────────────────────────────────────────────────

def list_presets() -> list[str]:
    """Return sorted list of preset names (stem only) from the presets directory."""
    if not PRESET_DIR.exists():
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in PRESET_DIR.glob("*.json"))
    return names if names else ["< default >"]


def preset_path(name: str) -> Path:
    """Return the full path for a preset name."""
    clean = name.strip().strip("<>").strip()
    if clean == "default" or not clean:
        clean = "default"
    return PRESET_DIR / f"{clean}.json"


def save_config_to_file(config: TrainConfig, path) -> None:
    """Serialize a TrainConfig to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(config.to_pack_dict(), f, indent=4)


def load_config_from_file(path) -> TrainConfig:
    """Deserialize a TrainConfig from a JSON file."""
    p = Path(path)
    with open(p, "r") as f:
        data = json.load(f)
    config = TrainConfig.default_values()
    config.from_pack_dict(data)
    return config


# ── Concept file helpers ────────────────────────────────────────────────────

def save_concepts_to_file(concept_dicts: list[dict], path: str) -> None:
    """Write a list of concept dicts to the concept JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(concept_dicts, f, indent=4)


def load_concepts_from_file(path: str) -> list[dict]:
    """Read concept JSON, return list of raw dicts."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def ensure_concept_file_exists(concept_file_name: str) -> None:
    """Create an empty concept JSON file if it doesn't exist yet."""
    p = Path(concept_file_name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump([], f)


# ── Sample file helpers ─────────────────────────────────────────────────────

def save_samples_to_file(sample_dicts: list[dict], path: str) -> None:
    """Write a list of sample dicts to the sample definition JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(sample_dicts, f, indent=4)


def load_samples_from_file(path: str) -> list[dict]:
    """Read sample definition JSON, return list of raw dicts."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def ensure_sample_file_exists(sample_file_name: str) -> None:
    """Create an empty sample definition JSON file if it doesn't exist yet."""
    p = Path(sample_file_name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump([], f)


# ── Dataset image preview helpers ───────────────────────────────────────────

def get_dataset_images(dataset_path: str, include_subdirs: bool = False) -> list[str]:
    """Return sorted list of supported image paths in a dataset directory.

    Excludes mask and condition label images (ending in -masklabel.png, -condlabel.png).
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        return []

    pattern = "**/*.*" if include_subdirs else "*.*"
    result = []
    for p in sorted(pathlib.Path(dataset_path).glob(pattern)):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if p.name.endswith("-masklabel.png") or p.name.endswith("-condlabel.png"):
            continue
        result.append(str(p))
    return result


def load_image_thumbnail(image_path: str, size: int = 200):
    """Load an image, crop to square, resize, and return as PIL Image.

    Returns PIL Image or None. Returning a PIL Image (not a path) avoids
    Gradio's InvalidPathError for files outside the working directory.
    """
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        img = PILImage.open(image_path).convert("RGB")
        # Crop to square (center crop)
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), PILImage.Resampling.BILINEAR)
        return img
    except (OSError, ValueError):
        return None


def get_image_prompt(image_path: str, prompt_source: str = "sample",
                     prompt_path: str = "") -> str:
    """Read prompt text for an image based on prompt source setting.

    Matches desktop ConceptWindow._read_text_file_for_preview logic.
    """
    if not image_path:
        return ""

    p = pathlib.Path(image_path)

    if prompt_source == "filename":
        return p.stem or ""

    if prompt_source == "concept":
        # Read from a single text file
        if prompt_path and os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except (OSError, UnicodeDecodeError):
                return "[Error reading prompt file]"
        return ""

    # Default: "sample" — read from per-image .txt file
    txt_path = p.with_suffix(".txt")
    if txt_path.exists():
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, UnicodeDecodeError):
            return "[Error reading prompt file]"
    return ""
