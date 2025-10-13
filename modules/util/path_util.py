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


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
SUPPORTED_VIDEO_EXTENSIONS = {'.webm', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.mp4', '.mpeg', '.m4v'}


def supported_image_extensions() -> set[str]:
    return SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def supported_video_extensions() -> set[str]:
    return SUPPORTED_VIDEO_EXTENSIONS


def is_supported_video_extension(extension: str) -> bool:
    return extension.lower() in SUPPORTED_VIDEO_EXTENSIONS


def validate_file_path(
    value: str,
    is_output: bool = False,
    valid_extensions: list[str] | None = None,
    path_type: str = "file",  # "file" or "directory"
) -> tuple[bool, str]:
    if not value:
        return True, ""

    # windows invalid characters
    if is_output and os.name == 'nt':
        invalid_chars = ['<', '>', '|', '\0', '"']
        path_parts = [os.path.basename(value)]

        dirname = os.path.dirname(value)
        if dirname:
            path_parts.extend(
                part for part in dirname.replace('\\', '/').split('/')
                if part and not (len(part) == 2 and part[1] == ':')
            )

        for part in path_parts:
            for char in invalid_chars:
                if char in part:
                    location = "filename" if part == path_parts[0] else "path"
                    return False, f"Invalid character in {location}: '{char}'"

    if is_output:
        if path_type == "directory" and os.path.isfile(value):
            return False, "Must be a directory, not a file"

        if path_type == "file":
            if os.path.isdir(value):
                return False, "Must be a file, not a directory"

            # Validate extension if applicable
            if valid_extensions and "." in os.path.basename(value):
                file_ext = os.path.splitext(value)[1].lower()
                if file_ext not in valid_extensions:
                    return False, f"Invalid extension. Expected: {', '.join(valid_extensions)}"

            # Validate parent directory exists
            parent_dir = os.path.dirname(value)
            if parent_dir and not os.path.isdir(parent_dir):
                return False, "Parent directory does not exist"

        return True, ""

    if path_type == "directory":
        if os.path.isfile(value) or (not os.path.isdir(value) and os.path.splitext(value)[1]):
            value = os.path.dirname(value) or '.'

        if not os.path.isdir(value):
            return False, "Directory does not exist"
    else:  # file
        if os.path.isdir(value):
            return False, "Must be a file, not a directory"
        if not os.path.isfile(value):
            return False, "File does not exist"

    return True, ""
