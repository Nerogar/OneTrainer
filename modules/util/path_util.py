import json
import os.path
from pathlib import Path
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


def json_path_modifier(x: str | Path) -> Path:
    x = Path(x).absolute()
    return x.parent if x.suffix == ".json" else x


MASK_POSTFIX = '-masklabel'
MASK_EXTENSION = '.png'
MAX_MASK_VARIANTS = 9

# Stem postfixes of every mask sidecar: the original '-masklabel' plus the
# numbered variants '-masklabel1' to '-masklabel9'. Listed explicitly because
# the dataset enumeration filter matches literal postfixes.
MASK_POSTFIXES = (MASK_POSTFIX,) + tuple(
    f'{MASK_POSTFIX}{variant}' for variant in range(1, MAX_MASK_VARIANTS + 1)
)


def mask_postfix(variant: int = 0) -> str:
    # variant 0 is the original '-masklabel', keeping existing datasets working
    return MASK_POSTFIX if variant == 0 else f'{MASK_POSTFIX}{variant}'


def mask_path_for(image_path: str, variant: int = 0) -> str:
    return os.path.splitext(image_path)[0] + mask_postfix(variant) + MASK_EXTENSION


def is_mask_filename(filename: str) -> bool:
    # matches '<name>-masklabel.png' and '<name>-masklabel1.png' .. '-masklabel9.png'.
    # tests the stem, the same rule the dataset enumeration uses to keep mask
    # sidecars out of the training images
    return os.path.splitext(filename)[0].endswith(MASK_POSTFIXES)
