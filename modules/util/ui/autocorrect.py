import contextlib
import os
import re
import sys
import unicodedata

from modules.util.enum.PathIOType import PathIOType

_INVISIBLE_UNICODE_CHARS_RE = re.compile(
    "["
    "\u200b"       # ZERO WIDTH SPACE
    "\u200c"       # ZERO WIDTH NON-JOINER
    "\u200d"       # ZERO WIDTH JOINER
    "\u200e"       # LEFT-TO-RIGHT MARK
    "\u200f"       # RIGHT-TO-LEFT MARK
    "\u2060"       # WORD JOINER
    "\ufeff"       # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u202a-\u202e"  # LRE, RLE, PDF, LRO, RLO
    "\u2066-\u2069"  # LRI, RLI, FSI, PDI
    "]"
)
_CONSECUTIVE_WHITESPACE_RE = re.compile(r"[ \t]+")
_NUMERIC_SEPARATOR_RE = re.compile(r"[_,\u066b]")
_NON_FLOAT_NOTATION_CHARS_RE = re.compile(r"[^0-9eE.\-]")
# Natch one decimal separator: period, comma, or Arabic momayyez (٫)
_ANY_DECIMAL_RE = re.compile(r"^(-?\d+)[.,\u066b](\d*)$")
_FOREIGN_DECIMAL_RE = re.compile(r"^(-?\d+)[,\u066b](\d+)$")
_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:[/\\]")
_FILE_EXTENSION_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]+$")


_IS_WINDOWS = sys.platform == "win32"

INVALID_PATH_CHARS: frozenset[str] = frozenset(chr(c) for c in range(32))
if _IS_WINDOWS:
    INVALID_PATH_CHARS = INVALID_PATH_CHARS | frozenset('<>"|?*')

_LEARNING_RATE_NON_VALUE_SUFFIXES = frozenset({
    "learning_rate_scheduler",
    "learning_rate_scaler",
    "learning_rate_cycles",
    "learning_rate_warmup_steps",
    "learning_rate_min_factor",
})


def _strip_matched_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def autocorrect_string(value: str) -> str:
    if not value:
        return value
    result = value.strip()
    result = _INVISIBLE_UNICODE_CHARS_RE.sub("", result)
    result = _CONSECUTIVE_WHITESPACE_RE.sub(" ", result)
    result = unicodedata.normalize("NFKC", result)
    return result


def autocorrect_int(value: str) -> str:
    if not value:
        return value
    result = autocorrect_string(value).lstrip("+").replace("_", "")
    # strip spaces between digits
    result = re.sub(r"(?<=\d) +(?=\d)", "", result)
    # if the value is only digits + exactly one decimal separator (. , ٫), round.
    if m := _ANY_DECIMAL_RE.match(result):
        # Exactly 3 digits after the separator ⇒ thousand separator, not decimal
        if len(m.group(2)) == 3:
            result = m.group(1) + m.group(2)
        else:
            with contextlib.suppress(ValueError):
                result = str(round(float(f"{m.group(1)}.{m.group(2)}")))
    else:
        result = _NUMERIC_SEPARATOR_RE.sub("", result)
    return result


def autocorrect_float(value: str, *, is_learning_rate: bool = False) -> str:
    if not value:
        return value

    result = autocorrect_string(value)

    if is_learning_rate:
        result = _NON_FLOAT_NOTATION_CHARS_RE.sub("", result)
        if not result:
            return result

    result = result.lstrip("+").replace("_", "")

    # Coerce a foreign decimal separator (comma, Arabic ٫) into a period
    if m := _FOREIGN_DECIMAL_RE.match(result):
        result = f"{m.group(1)}.{m.group(2)}"
    else:
        result = _NUMERIC_SEPARATOR_RE.sub("", result)

    result = re.sub(r"^(-?)\.", r"\g<1>0.", result)
    if result.endswith(".") and len(result) > 1:
        result += "0"

    return result


def autocorrect_path(
    value: str,
    io_type: PathIOType = PathIOType.INPUT,
    expected_ext: str | None = None,
) -> str:
    if not value:
        return value

    result = _strip_matched_quotes(value.strip())
    if not result:
        return result

    # Silently strip characters that are never valid in a path.
    result = result.translate({ord(c): None for c in INVALID_PATH_CHARS})
    if not result:
        return result

    if result.startswith("~"):
        result = os.path.expanduser(result)

    if _IS_WINDOWS:
        is_unc = result.startswith(("\\\\", "//"))
        result = os.path.normpath(result.replace("/", "\\"))
        if is_unc and not result.startswith("\\\\"):
            result = "\\\\" + result.lstrip("\\")
    elif not _WINDOWS_DRIVE_PREFIX_RE.match(result):
        result = os.path.normpath(result)

    if io_type in (PathIOType.OUTPUT, PathIOType.MODEL):
        result = result.rstrip(os.sep)

    if expected_ext is not None and io_type == PathIOType.MODEL and result:
        result = _FILE_EXTENSION_SUFFIX_RE.sub("", result) + expected_ext

    return result


def is_learning_rate_field(var_name: str) -> bool:
    """True if *var_name* is a learning-rate value field (excludes scheduler/scaler/etc)."""
    leaf = var_name.rsplit(".", 1)[-1]
    if leaf in _LEARNING_RATE_NON_VALUE_SUFFIXES:
        return False
    return "learning_rate" in leaf
