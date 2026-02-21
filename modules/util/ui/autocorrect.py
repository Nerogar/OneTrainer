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
    result = value.strip()
    if result.startswith("+") and len(result) > 1:
        result = result[1:]
    result = result.replace("_", "")
    # strip spaces between digits
    result = re.sub(r"(?<=\d) +(?=\d)", "", result)
    # if the value is only digits + exactly one decimal separator (. , ٫), round.
    m = _ANY_DECIMAL_RE.match(result)
    if m:
        # Exactly 3 digits after the separator ⇒ thousand separator, not decimal
        if len(m.group(2)) == 3:
            result = m.group(1) + m.group(2)
        else:
            normalized = m.group(1) + "." + m.group(2)
            with contextlib.suppress(ValueError):
                result = str(round(float(normalized)))
    else:
        result = _NUMERIC_SEPARATOR_RE.sub("", result)
    return result


def autocorrect_float(value: str, *, is_learning_rate: bool = False) -> str:
    if not value:
        return value

    result = value.strip()

    if is_learning_rate:
        result = _NON_FLOAT_NOTATION_CHARS_RE.sub("", result)
        if not result:
            return result

    if result.startswith("+") and len(result) > 1:
        result = result[1:]

    result = result.replace("_", "")
    # Coerce a foreign decimal separator (comma, Arabic ٫) into a period
    m = _FOREIGN_DECIMAL_RE.match(result)
    result = m.group(1) + "." + m.group(2) if m else _NUMERIC_SEPARATOR_RE.sub("", result)

    if result.startswith("."):
        result = "0" + result
    elif result.startswith("-.") and len(result) > 2:
        result = "-0" + result[1:]

    if result.endswith(".") and len(result) > 1:
        result = result + "0"

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
    if INVALID_PATH_CHARS.intersection(result):
        result = "".join(c for c in result if c not in INVALID_PATH_CHARS)
        if not result:
            return result

    if result.startswith("~"):
        result = os.path.expanduser(result)

    if _IS_WINDOWS:
        is_unc = result.startswith(("\\\\", "//"))

        if _WINDOWS_DRIVE_PREFIX_RE.match(result) or is_unc:
            result = result.replace("/", "\\")

        normalised = os.path.normpath(result)

        if is_unc and not normalised.startswith("\\\\"):
            normalised = "\\\\" + normalised.lstrip("\\")

        result = normalised
    else:
        if not _WINDOWS_DRIVE_PREFIX_RE.match(result):
            result = os.path.normpath(result)

    if io_type in (PathIOType.OUTPUT, PathIOType.MODEL):
        result = result.rstrip(os.sep)

    if expected_ext is not None and io_type == PathIOType.MODEL and result:
        existing_ext_match = _FILE_EXTENSION_SUFFIX_RE.search(result)
        if expected_ext == "":
            # Diffusers: directory output — strip any file extension
            if existing_ext_match:
                result = result[: existing_ext_match.start()]
        elif existing_ext_match:
            if existing_ext_match.group().lower() != expected_ext.lower():
                result = result[: existing_ext_match.start()] + expected_ext
        else:
            result += expected_ext

    return result


def is_learning_rate_field(var_name: str) -> bool:
    """True if *var_name* is a learning-rate value field (excludes scheduler/scaler/etc)."""
    leaf = var_name.rsplit(".", 1)[-1]
    if leaf in _LEARNING_RATE_NON_VALUE_SUFFIXES:
        return False
    return "learning_rate" in leaf
