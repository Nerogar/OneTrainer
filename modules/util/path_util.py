import json
import os.path
from typing import Any


def safe_filename(
        text: str,
        allow_spaces: bool = True,
        max_length: int | None = 32,
):
    legal_chars = [' ', '.', '_', '-', '#']
    if not allow_spaces:
        text = text.replace(' ', '_')

    text = ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text)).strip()

    if max_length is not None:
        text = text[0: max_length]

    return text.strip()


def canonical_join(base_path: str, *paths: str):
    # Creates a canonical path name that can be used for comparisons.
    # Also, Windows does understand / instead of \, so these paths can be used as usual.

    joined = os.path.join(base_path, *paths)
    return joined.replace('\\', '/')


def write_json_atomic(path: str, obj: Any):
    with open(path + ".write", "w") as f:
        json.dump(obj, f, indent=4)
    os.replace(path + ".write", path)


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.avif'}
SUPPORTED_VIDEO_EXTENSIONS = {'.webm', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.mp4', '.mpeg', '.m4v'}
SUPPORTED_CAPTION_EXTENSIONS = {'.txt'}


def supported_image_extensions() -> set[str]:
    return SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> set[str]:
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_video_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


def supported_caption_extensions() -> set[str]:
    return SUPPORTED_CAPTION_EXTENSIONS


def walk_skipping_dotted(folder: str, include_subdirectories: bool = True):
    """``os.walk`` wrapper that prunes dot-prefixed subdirectories in-place
    (``.thumbnails``, ``.cache``, ``.stversions``, ...) so their contents are
    never yielded — gallery apps and sync tools plant resized previews there
    that would otherwise be picked up as real images, the same as mgds
    CollectPaths. Yields ``(root, files)`` pairs. The top-level ``folder`` itself
    is always walked, even when its own name starts with a dot. When
    ``include_subdirectories`` is False only the top level's files are yielded.
    A missing ``folder`` yields nothing rather than raising."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = [] if not include_subdirectories else [d for d in dirs if not d.startswith(".")]
        yield root, files
