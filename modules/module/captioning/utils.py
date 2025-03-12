import re
from pathlib import Path

from modules.util import path_util


def is_supported_image(path: Path) -> bool:
    """Returns True if the file has a supported image extension and is not a mask label image."""
    return path_util.is_supported_image_extension(path.suffix) and "-masklabel.png" not in path.name


def get_sample_filenames(sample_dir: str, include_subdirs: bool = False) -> list[str]:
    sample_path = Path(sample_dir)
    pattern = "**/*" if include_subdirs else "*"
    return [str(p) for p in sample_path.glob(pattern) if p.is_file() and is_supported_image(p)]


def is_empty_caption(caption: str) -> bool:
    # Check effectively empty captions.
    # Removes whitespace and punctuation that doesn't affect content.
    stripped = re.sub(r"[\s\.,_`();:'\"-]+", "", caption)
    return not stripped
