from __future__ import annotations

import contextlib
import os
import re
import sys
import tkinter as tk
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from modules.util.enum.PathIOType import PathIOType

if TYPE_CHECKING:
    from modules.util.ui.UIState import UIState

    import customtkinter as ctk


DEBOUNCE_TYPING_MS = 250
UNDO_DEBOUNCE_MS = 500
ERROR_BORDER_COLOR = "#dc3545"

_active_validators: set[FieldValidator] = set()

TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")
HUGGINGFACE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

_INVALID_CHARS = {chr(c) for c in range(32)}
_IS_WINDOWS = sys.platform == "win32"
if _IS_WINDOWS:
    _INVALID_CHARS |= set('<>"|?*')

_FORMAT_EXTENSIONS: dict[str, str] = {
    "CKPT": ".ckpt",
    "SAFETENSORS": ".safetensors",
    "LEGACY_SAFETENSORS": ".safetensors",
    "COMFY_LORA": ".safetensors",
}


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

    if io_type == PathIOType.INPUT and _is_huggingface_repo_or_file(trimmed):
        return None

    if io_type in (PathIOType.OUTPUT, PathIOType.MODEL):
        if not os.path.isdir(os.path.dirname(os.path.abspath(trimmed))):
            return "Parent folder does not exist"

    if io_type == PathIOType.MODEL and output_format is not None:
        if output_format == "DIFFUSERS":
            if ENDS_WITH_EXT.search(trimmed):
                return "Diffusers output must be a directory path, not a file"
            return _check_overwrite(trimmed, is_dir=True, prevent=prevent_overwrites)

        expected_ext = _FORMAT_EXTENSIONS.get(output_format, "")
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


class DebounceTimer:
    def __init__(self, widget, delay_ms: int, callback: Callable[..., Any]):
        self.widget = widget
        self.delay_ms = delay_ms
        self.callback = callback
        self._after_id: str | None = None

    def call(self, *args, **kwargs):
        if self._after_id:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)

        def fire():
            self._after_id = None
            self.callback(*args, **kwargs)

        with contextlib.suppress(tk.TclError):
            self._after_id = self.widget.after(self.delay_ms, fire)

    def cancel(self):
        if self._after_id:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)
            self._after_id = None


@dataclass
class _ValidationState:
    touched: bool = False
    trace_name: str | None = None
    bound: bool = False


class FieldValidator:
    def __init__(
        self,
        component: ctk.CTkEntry,
        var: tk.Variable,
        ui_state: UIState,
        var_name: str,
        max_undo: int = DEFAULT_MAX_UNDO,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        self.component = component
        self.var = var
        self.ui_state = ui_state
        self.var_name = var_name
        self._extra_validate = extra_validate
        self._required = required

        try:
            self._original_border_color = component.cget("border_color")
        except Exception:
            self._original_border_color = "gray50"

        self._state = _ValidationState()
        self._debounce: DebounceTimer | None = None
        self._undo_debounce: DebounceTimer | None = None
        self._undo = UndoHistory(max_undo)
        self._applying_undo = False

    def attach(self) -> None:
        s = self._state
        self._debounce = DebounceTimer(
            self.component, DEBOUNCE_TYPING_MS, self._on_debounce_fire
        )
        self._undo_debounce = DebounceTimer(
            self.component, UNDO_DEBOUNCE_MS, self._push_undo_snapshot
        )

        s.trace_name = self.var.trace_add("write", self._on_var_write)

        self.component.bind("<FocusIn>", self._on_focus_in)
        self.component.bind("<Key>", self._on_user_input)
        self.component.bind("<<Paste>>", self._on_user_input)
        self.component.bind("<<Cut>>", self._on_user_input)
        self.component.bind("<FocusOut>", self._on_focus_out)
        self.component.bind("<Control-z>", self._on_undo)
        self.component.bind("<Control-Z>", self._on_undo)
        self.component.bind("<Control-Shift-z>", self._on_redo)
        self.component.bind("<Control-Shift-Z>", self._on_redo)
        self.component.bind("<Control-y>", self._on_redo)
        self.component.bind("<Control-Y>", self._on_redo)

        s.bound = True
        _active_validators.add(self)

    def detach(self) -> None:
        s = self._state
        if not s.bound:
            return
        s.bound = False
        _active_validators.discard(self)

        if self._debounce:
            self._debounce.cancel()
        if self._undo_debounce:
            self._undo_debounce.cancel()

        if s.trace_name:
            with contextlib.suppress(Exception):
                self.var.trace_remove("write", s.trace_name)
            s.trace_name = None

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
                if v < 0:
                    return "Value must be non-negative"
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

    def _apply_error(self) -> None:
        self.component.configure(border_color=ERROR_BORDER_COLOR)

    def _clear_error(self) -> None:
        self.component.configure(border_color=self._original_border_color)

    def _validate_and_style(self, value: str) -> bool:
        error = self.validate(value)
        if error is None:
            self._clear_error()
            return True
        else:
            self._apply_error()
            return False

    def _on_var_write(self, *_args) -> None:
        if self._applying_undo:
            return
        s = self._state
        if not s.touched:
            if self._debounce:
                self._debounce.cancel()
            return
        if self._debounce:
            self._debounce.call()
        if self._undo_debounce:
            self._undo_debounce.call()

    def _push_undo_snapshot(self) -> None:
        self._undo.push(self.var.get())

    def _on_debounce_fire(self) -> None:
        self._validate_and_style(self.var.get())

    def _on_focus_in(self, _e=None) -> None:
        self._state.touched = False
        self._undo.push(self.var.get())

    def _on_user_input(self, _e=None) -> None:
        self._state.touched = True

    def _on_focus_out(self, _e=None) -> None:
        if self._state.touched:
            self._validate_and_style(self.var.get())
        if self._undo_debounce:
            self._undo_debounce.cancel()
        self._undo.push(self.var.get())

    def _set_value(self, value: str) -> None:
        self._applying_undo = True
        self.var.set(value)
        self._applying_undo = False
        self._validate_and_style(value)

    def _on_undo(self, _e=None) -> str:
        previous = self._undo.undo(self.var.get())
        if previous is not None:
            self._set_value(previous)
        return "break"

    def _on_redo(self, _e=None) -> str:
        next_val = self._undo.redo()
        if next_val is not None:
            self._set_value(next_val)
        return "break"


class PathValidator(FieldValidator):
    """FieldValidator with additional path-specific checks."""

    def __init__(
        self,
        component: ctk.CTkEntry,
        var: tk.Variable,
        ui_state: UIState,
        var_name: str,
        io_type: PathIOType = PathIOType.INPUT,
        max_undo: int = DEFAULT_MAX_UNDO,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(component, var, ui_state, var_name, max_undo=max_undo, extra_validate=extra_validate, required=required)
        self.io_type = io_type

    def _get_var_safe(self, name: str) -> tk.Variable | None:
        try:
            return self.ui_state.get_var(name)
        except (KeyError, AttributeError):
            return None

    def validate(self, value: str) -> str | None:
        base_err = super().validate(value)
        if base_err is not None:
            return base_err
        if value == "":
            return None

        prevent_var = self._get_var_safe("prevent_overwrites")
        format_var = self._get_var_safe("output_model_format")
        return validate_path(
            value,
            io_type=self.io_type,
            prevent_overwrites=prevent_var.get() if prevent_var is not None else False,
            output_format=format_var.get() if format_var is not None else None,
        )

    def revalidate(self) -> None:
        if self.component.winfo_exists():
            self._validate_and_style(self.var.get())


def flush_and_validate_all() -> list[str]:
    invalid: list[str] = []

    for v in list(_active_validators):
        if v._debounce:
            v._debounce.cancel()

        value = v.var.get()
        error = v.validate(value)

        if error is not None:
            v._apply_error()
            invalid.append(f"{v.var_name}: {error}")
        else:
            v._clear_error()

    return invalid
