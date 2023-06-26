import os.path


def safe_filename(text: str):
    legal_chars = [' ', '.', '_', '-', '#']
    return ''.join(filter(lambda x: str.isalnum(x) or x in legal_chars, text))[0:32].strip()


def canonical_join(base_path: str, *paths: str):
    # Creates a canonical path name that can be used for comparisons.
    # Also, Windows does understand / instead of \, so these paths can be used as usual.

    joined = os.path.join(base_path, *paths)
    return joined.replace('\\', '/')


def supported_image_extensions() -> list[str]:
    return ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']


def is_supported_image_extension(extension: str) -> bool:
    return extension.lower() in supported_image_extensions()
