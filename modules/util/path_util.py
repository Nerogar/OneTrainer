import os.path


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
