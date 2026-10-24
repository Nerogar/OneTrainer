from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType

if TYPE_CHECKING:
    from modules.util.ui.UIState import BaseUIState


DEBOUNCE_TYPING_MS = 250
UNDO_DEBOUNCE_MS = 500
ERROR_BORDER_COLOR = "#dc3545"

TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")
HUGGINGFACE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

_INVALID_CHARS = {chr(c) for c in range(32)}
_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    _INVALID_CHARS |= set('<>"|?*')


def _is_huggingface_repo_or_file(value: str) -> bool:
    trimmed = value.strip()

    if trimmed.startswith("https://"):
        parsed = urlparse(trimmed)
        if parsed.netloc not in {"huggingface.co", "huggingface.com"}:
            return False
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"resolve", "blob"}:
            return bool(ENDS_WITH_EXT.search(parts[-1]))
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

    return bool(HUGGINGFACE_REPO_RE.match(trimmed))


def _has_invalid_chars(value: str) -> bool:
    return bool(_INVALID_CHARS.intersection(value))


def _check_overwrite(path: str, *, is_dir: bool, prevent: bool) -> str | None:
    if not prevent:
        return None
    abs_path = os.path.abspath(path)
    if is_dir and os.path.isdir(abs_path):
        return "Output folder already exists (overwrite prevented)"
    if not is_dir and os.path.isfile(abs_path):
        return "Output file already exists (overwrite prevented)"
    return None


def validate_path(
    value: str,
    io_type: PathIOType = PathIOType.INPUT,
    *,
    prevent_overwrites: bool = False,
    output_format: str | None = None,
) -> str | None:
    """Return an error string if *value* is an invalid path, else ``None``."""
    trimmed = value.strip()

    if not trimmed:
        return "Path is empty"
    if TRAILING_SLASH_RE.search(trimmed):
        return "Path must not end with a slash"
    if _has_invalid_chars(trimmed):
        return "Path contains invalid characters"

    if trimmed.startswith("cloud:"):
        cloud_path = trimmed[6:]
        if not cloud_path:
            return "Cloud path is empty"
        if cloud_path.startswith(("http://", "https://")):
            return "Cloud path cannot be a URL"
        if "\\" in cloud_path:
            return "Cloud path must use forward slashes (/)"
        return None

    if io_type == PathIOType.INPUT and _is_huggingface_repo_or_file(trimmed):
        return None

    if io_type == PathIOType.INPUT:
        if not os.path.exists(os.path.abspath(trimmed)):
            return "Input path does not exist"

    if io_type in (PathIOType.OUTPUT, PathIOType.MODEL):
        if not os.path.isdir(os.path.dirname(os.path.abspath(trimmed))):
            return "Parent folder does not exist"

    if io_type == PathIOType.MODEL and output_format is not None:
        if output_format == "DIFFUSERS":
            if ENDS_WITH_EXT.search(trimmed):
                return "Diffusers output must be a directory path, not a file"
            return _check_overwrite(trimmed, is_dir=True, prevent=prevent_overwrites)

        try:
            expected_ext = ModelFormat[output_format].file_extension()
        except KeyError:
            expected_ext = ""

        if expected_ext:
            suffix = (PureWindowsPath(trimmed) if _IS_WINDOWS else PurePosixPath(trimmed)).suffix.lower()
            if suffix != expected_ext:
                return f"Extension must be '{expected_ext}' for {output_format} format"
        return _check_overwrite(trimmed, is_dir=False, prevent=prevent_overwrites)

    if io_type == PathIOType.OUTPUT:
        return _check_overwrite(trimmed, is_dir=False, prevent=prevent_overwrites)

    return None

DEFAULT_MAX_UNDO = 20


class UndoHistory:
    def __init__(self, max_size: int = DEFAULT_MAX_UNDO):
        self._stack: deque[str] = deque(maxlen=max_size)
        self._redo_stack: list[str] = []

    def push(self, value: str):
        if self._stack and self._stack[-1] == value:
            return
        self._stack.append(value)
        self._redo_stack.clear()

    def undo(self, current: str) -> str | None:
        if not self._stack:
            return None
        top = self._stack[-1]
        if top == current and len(self._stack) > 1:
            self._redo_stack.append(self._stack.pop())
            return self._stack[-1]
        elif top != current:
            self._redo_stack.append(current)
            return top
        return None

    def redo(self) -> str | None:
        if not self._redo_stack:
            return None
        value = self._redo_stack.pop()
        self._stack.append(value)
        return value


class BaseFieldValidator(ABC):
    def __init__(
        self,
        ui_state: BaseUIState,
        var_name: str,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        self.ui_state = ui_state
        self.var_name = var_name
        self._extra_validate = extra_validate
        self._required = required
        self._bound = False

    @abstractmethod
    def _apply_error(self) -> None:
        pass

    @abstractmethod
    def _clear_error(self) -> None:
        pass

    @abstractmethod
    def flush(self) -> str | None:
        pass

    def _get_var_safe(self, name: str):
        try:
            return self.ui_state.get_var(name)
        except (KeyError, AttributeError):
            return None

    def validate(self, value: str) -> str | None:
        """Return an error string if *value* is invalid, else None."""
        meta = self.ui_state.get_field_metadata(self.var_name)
        declared_type = meta.type
        nullable = meta.nullable
        default_val = meta.default

        if value == "":
            if self._required:
                return "Value required"
            if nullable:
                return None
            if declared_type is str:
                if default_val == "":
                    return None
                return "Value required"
            return None

        try:
            if declared_type is int:
                v = int(value)
            elif declared_type is float:
                v = float(value)
                if v < 0:
                    return "Value must be non-negative"
            elif declared_type is bool:
                if value.lower() not in ("true", "false", "0", "1"):
                    return "Invalid bool"
        except ValueError:
            return "Invalid value"

        if self._extra_validate is not None:
            return self._extra_validate(value)

        return None

    def _validate_and_style(self, value: str) -> bool:
        error = self.validate(value)
        if error is None:
            self._clear_error()
            return True
        else:
            self._apply_error()
            return False


_active_validators: set[BaseFieldValidator] = set()


def flush_and_validate_all() -> list[str]:
    invalid: list[str] = []
    for v in list(_active_validators):
        error = v.flush()
        if error is not None:
            invalid.append(f"{v.var_name}: {error}")
    return invalid


def _validate_path_field(ui_state: BaseUIState, io_type: PathIOType, value: str) -> str | None:
    try:
        prevent_var = ui_state.get_var("prevent_overwrites")
    except (KeyError, AttributeError):
        prevent_var = None
    try:
        format_var = ui_state.get_var("output_model_format")
    except (KeyError, AttributeError):
        format_var = None
    return validate_path(
        value,
        io_type=io_type,
        prevent_overwrites=prevent_var.get() if prevent_var is not None else False,
        output_format=format_var.get() if format_var is not None else None,
    )
