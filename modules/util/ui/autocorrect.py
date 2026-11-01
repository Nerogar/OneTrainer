import os
import re
import sys
from urllib.parse import urlparse

from modules.util.enum.PathIOType import PathIOType

_ENDS_WITH_EXT_RE = re.compile(r"\.[A-Za-z0-9]+$")
_HUGGINGFACE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


_IS_WINDOWS = sys.platform == "win32"

INVALID_PATH_CHARS: frozenset[str] = frozenset(chr(c) for c in range(32))

if _IS_WINDOWS:
    INVALID_PATH_CHARS = INVALID_PATH_CHARS | frozenset('<>"|?*')


def _strip_matched_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def is_huggingface_repo_or_file(value: str) -> bool:
    trimmed = value.strip()

    if trimmed.startswith("https://"):
        parsed = urlparse(trimmed)
        if parsed.netloc not in {"huggingface.co", "huggingface.com"}:
            return False
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"resolve", "blob"}:
            return bool(_ENDS_WITH_EXT_RE.search(parts[-1]))
        return False

    if len(trimmed) > 96:
        return False
    if " " in trimmed or "\t" in trimmed:
        return False
    if "—" in trimmed or ".." in trimmed:
        return False
    if trimmed.startswith(("\\\\", "//", "/")):
        return False
    if len(trimmed) >= 2 and trimmed[1] == ":" and trimmed[0].isalpha():
        return False
    if trimmed.count("/") != 1:
        return False

    return bool(_HUGGINGFACE_REPO_RE.match(trimmed))


def autocorrect_string(value: str) -> str:
    """Trim surrounding whitespace without changing the contents."""
    return value.strip()


def autocorrect_int(value: str) -> str:
    return value.strip()


def autocorrect_float(value: str) -> str:
    return value.strip()


def autocorrect_path(
    value: str,
    io_type: PathIOType = PathIOType.INPUT,
) -> str:
    """Clean pasted path syntax without changing path components.

    Input and output paths deliberately use the same cleanup rules.
    """
    if not value:
        return value

    result = _strip_matched_quotes(value.strip())
    if not result:
        return result

    if result.startswith("cloud:") or is_huggingface_repo_or_file(result):
        return result

    # Extended Windows paths have literal path semantics.
    if result.startswith(("\\\\?\\", "\\\\.\\")):
        return result

    if result.startswith("~"):
        result = os.path.expanduser(result)

    if _IS_WINDOWS:
        result = result.replace("/", "\\")

    return result
