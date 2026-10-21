from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType
from modules.util.ui.autocorrect import INVALID_PATH_CHARS, is_huggingface_repo_or_file

if TYPE_CHECKING:
    from modules.util.ui.UIState import BaseUIState


DEBOUNCE_TYPING_MS = 250
UNDO_DEBOUNCE_MS = 500
ERROR_BORDER_COLOR = "#dc3545"
_MAX_DISPLAY_CHARS = 5

TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")
_IS_WINDOWS = sys.platform == "win32"


def _format_char(c: str) -> str:
    """Return a human-readable representation of a single character."""
    cp = ord(c)
    if cp < 32:
        return f"U+{cp:04X}" # control chars
    # Printable but forbidden (e.g. Windows-reserved)
    return repr(c)


def _describe_invalid_chars(value: str) -> str:
    """Return a suffix like ``': '?', '*'`` listing the offending characters."""
    bad = sorted(set(value) & INVALID_PATH_CHARS)
    if not bad:
        return ""

    shown = ", ".join(_format_char(c) for c in bad[:_MAX_DISPLAY_CHARS])
    if len(bad) > _MAX_DISPLAY_CHARS:
        shown += f" and {len(bad) - _MAX_DISPLAY_CHARS} more"
    return f": {shown}"


def _has_invalid_chars(value: str) -> bool:
    return bool(INVALID_PATH_CHARS.intersection(value))


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
    output_is_dir: bool = False,
    output_format: str | None = None,
) -> str | None:
    """Return an error string if *value* is an invalid path, else ``None``."""
    if not value:
        return "Path is empty"
    # The '?' in the Windows extended-path prefix is not a filename character.
    path_chars = value[4:] if _IS_WINDOWS and value.startswith("\\\\?\\") else value
    if _has_invalid_chars(path_chars):
        return "Path contains invalid characters" + _describe_invalid_chars(path_chars)

    if value.startswith("cloud:"):
        cloud_path = value[6:]
        if not cloud_path:
            return "Cloud path is empty"
        if cloud_path.startswith(("http://", "https://")):
            return "Cloud path cannot be a URL"
        if "\\" in cloud_path:
            return "Cloud path must use forward slashes (/)"
        return None

    if io_type == PathIOType.INPUT and is_huggingface_repo_or_file(value):
        return None

    if io_type == PathIOType.INPUT:
        if not os.path.exists(os.path.abspath(value)):
            return "Input path does not exist"

    if io_type == PathIOType.MODEL:
        fmt = ModelFormat.__members__.get(output_format)
        is_dir = fmt is not None and not fmt.is_single_file()
        if is_dir and ENDS_WITH_EXT.search(value):
            return "Diffusers output must be a directory path, not a file"
        if not is_dir and fmt is not None:
            expected = fmt.file_extension()
            suffix = (PureWindowsPath(value) if _IS_WINDOWS else PurePosixPath(value)).suffix.lower()
            if expected and suffix != expected:
                return f"Extension must be '{expected}' for {output_format} format"
        # A directory model is the artifact itself: validate only its parent.
        error = validate_path(os.path.dirname(os.path.abspath(value)), PathIOType.OUTPUT, output_is_dir=True) \
            if is_dir else validate_path(value, PathIOType.OUTPUT)
        return error or _check_overwrite(value, is_dir=is_dir, prevent=prevent_overwrites)

    if io_type == PathIOType.OUTPUT:
        if not output_is_dir and TRAILING_SLASH_RE.search(value):
            return "Output file path must not end with a slash"
        abs_path = os.path.abspath(value)
        output_dir = abs_path if output_is_dir else os.path.dirname(abs_path)
        while not os.path.exists(output_dir):
            parent = os.path.dirname(output_dir)
            if parent == output_dir:
                break
            output_dir = parent
        if os.path.exists(output_dir) and not os.path.isdir(output_dir):
            return f"Output folder is not a directory: {output_dir}"
        if not output_is_dir:
            return _check_overwrite(value, is_dir=False, prevent=prevent_overwrites)

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
                if v < 0 and self._extra_validate is None:
                    return "Value must be non-negative"
            elif declared_type is float:
                v = float(value)
                if v < 0 and self._extra_validate is None:
                    return "Value must be non-negative"
            elif declared_type is bool:
                if value.lower() not in ("true", "false", "0", "1"):
                    return "Invalid bool"
        except ValueError:
            if declared_type is int:
                return "Enter a whole number (for example, 2)"
            if declared_type is float:
                return "Enter a number using a decimal point (for example, 0.5 or 1e-4)"
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
    for v in sorted(_active_validators, key=lambda v: v.var_name == "run_name"):
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
