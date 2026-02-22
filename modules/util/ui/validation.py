from __future__ import annotations

import contextlib
import os
import re
import sys
import tkinter as tk
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType
from modules.util.ui.autocorrect import (
    INVALID_PATH_CHARS,
    autocorrect_float,
    autocorrect_int,
    autocorrect_path,
    autocorrect_string,
    is_learning_rate_field,
)
from modules.util.ui.ToolTip import ValidationTooltip

import friendlywords as fw

if TYPE_CHECKING:
    from modules.util.ui.UIState import UIState

    import customtkinter as ctk


DEBOUNCE_TYPING_MS = 500
UNDO_DEBOUNCE_MS = 500
ERROR_BORDER_COLOR = "#dc3545"

_active_validators: set[FieldValidator] = set()

TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")
HUGGINGFACE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

_IS_WINDOWS = sys.platform == "win32"

_MAX_DISPLAY_CHARS = 5


def _format_char(c: str) -> str:
    """Return a human-readable representation of a single character."""
    cp = ord(c)
    if cp < 32:
        # Control characters — show as U+00XX
        return f"U+{cp:04X}"
    # Printable but forbidden (e.g. Windows-reserved)
    return repr(c)


def _describe_invalid_chars(value: str) -> str:
    """Return a suffix like ``': '?', '*'`` listing the offending characters."""
    seen: set[str] = set()
    bad: list[str] = []
    for ch in value:
        if ch in INVALID_PATH_CHARS and ch not in seen:
            seen.add(ch)
            bad.append(ch)
    if not bad:
        return ""
    bad.sort(key=ord)
    shown = ", ".join(_format_char(c) for c in bad[:_MAX_DISPLAY_CHARS])
    extra = len(bad) - _MAX_DISPLAY_CHARS
    if extra > 0:
        shown += f" and {extra} more"
    return f": {shown}"


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
    return bool(INVALID_PATH_CHARS.intersection(value))


def _check_overwrite(path: str, *, is_dir: bool, prevent: bool) -> str | None:
    if not prevent:
        return None
    abs_path = os.path.abspath(path)
    check = os.path.isdir if is_dir else os.path.isfile
    if check(abs_path):
        kind = "folder" if is_dir else "file"
        return f"Output {kind} already exists (overwrite prevented)"
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
        return "Path contains invalid characters" + _describe_invalid_chars(trimmed)

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

        self._original_border_color = "gray50"
        with contextlib.suppress(Exception):
            self._original_border_color = component.cget("border_color")

        self._shadow_var = tk.StringVar(master=component)
        self._shadow_trace_name: str | None = None
        self._real_var_trace_name: str | None = None
        self._syncing = False
        self._touched = False
        self._bound = False

        self._debounce: DebounceTimer | None = None
        self._undo_debounce: DebounceTimer | None = None
        self._undo = UndoHistory(max_undo)
        self._tooltip: ValidationTooltip | None = None

    def _widget_alive(self) -> bool:
        try:
            return self.component.winfo_exists()
        except tk.TclError:
            return False

    def _get_var_safe(self, name: str) -> tk.Variable | None:
        try:
            return self.ui_state.get_var(name)
        except (KeyError, AttributeError):
            return None

    def _get_var_value(self, name: str, default: Any = None) -> Any:
        var = self._get_var_safe(name)
        return var.get() if var is not None else default

    def _cancel_debounces(self) -> None:
        if self._debounce:
            self._debounce.cancel()
        if self._undo_debounce:
            self._undo_debounce.cancel()

    def _auto_correct_enabled(self) -> bool:
        return bool(self._get_var_value("auto_correct_input", False))

    def _autocorrect_value(self, value: str) -> str:
        if not value:
            return value
        meta = self.ui_state.get_field_metadata(self.var_name)
        declared_type = meta.type
        if declared_type is int:
            return autocorrect_int(value)
        elif declared_type is float:
            return autocorrect_float(value, is_learning_rate=is_learning_rate_field(self.var_name))
        elif declared_type is str:
            return autocorrect_string(value)
        return value

    def _apply_autocorrect(self, value: str) -> str:
        if not self._auto_correct_enabled():
            return value
        corrected = self._autocorrect_value(value)
        if corrected != value:
            self._undo.push(value)
            self._syncing = True
            self._shadow_var.set(corrected)
            self._syncing = False
        return corrected

    def attach(self) -> None:
        self._shadow_var.set(self.var.get())
        self._swap_textvariable(self._shadow_var)

        self._debounce = DebounceTimer(
            self.component, DEBOUNCE_TYPING_MS, self._on_debounce_fire
        )
        self._undo_debounce = DebounceTimer(
            self.component, UNDO_DEBOUNCE_MS, self._push_undo_snapshot
        )

        self._shadow_trace_name = self._shadow_var.trace_add("write", self._on_shadow_write)
        self._real_var_trace_name = self.var.trace_add("write", self._on_real_var_write)

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
        self.component.bind("<Return>", self._on_enter)

        self._bound = True
        _active_validators.add(self)

    def detach(self) -> None:
        if not self._bound:
            return
        self._bound = False
        _active_validators.discard(self)

        self._commit()
        self._cancel_debounces()

        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None

        if self._shadow_trace_name:
            with contextlib.suppress(Exception):
                self._shadow_var.trace_remove("write", self._shadow_trace_name)
            self._shadow_trace_name = None

        if self._real_var_trace_name:
            with contextlib.suppress(Exception):
                self.var.trace_remove("write", self._real_var_trace_name)
            self._real_var_trace_name = None

        self._swap_textvariable(self.var)

    def _swap_textvariable(self, new_var: tk.Variable) -> None:
        comp = self.component
        if comp._textvariable_callback_name:
            with contextlib.suppress(Exception):
                comp._textvariable.trace_remove("write", comp._textvariable_callback_name)  # type: ignore[union-attr]
            comp._textvariable_callback_name = ""

        comp.configure(textvariable=new_var)

        if new_var is not None:
            # Wrap the CTkEntry callback so it won't fire on a destroyed widget
            original_cb = comp._textvariable_callback

            def _safe_textvariable_callback(*args):
                try:
                    if comp.winfo_exists() and comp._entry.winfo_exists():
                        original_cb(*args)
                except tk.TclError:
                    pass

            comp._textvariable_callback_name = new_var.trace_add(
                "write", _safe_textvariable_callback
            )

    def _commit(self) -> None:
        shadow_val = self._shadow_var.get()
        if shadow_val != self.var.get():
            self._syncing = True
            self.var.set(shadow_val)
            self._syncing = False

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
            if declared_type in (int, float):
                if declared_type(value) < 0:
                    return "Value must be non-negative"
                if self._required and declared_type is int and int(value) == 0:
                    return "Value must be greater than zero"
            elif declared_type is bool:
                if value.lower() not in ("true", "false", "0", "1"):
                    return "Invalid bool"
        except ValueError:
            return "Invalid value"

        if self._extra_validate is not None:
            return self._extra_validate(value)

        return None

    def _apply_error(self, error_msg: str = "", *, from_typing: bool = False) -> None:
        if not self._widget_alive():
            return
        self.component.configure(border_color=ERROR_BORDER_COLOR)
        if self._tooltip is None:
            self._tooltip = ValidationTooltip(self.component)
        self._tooltip.show_error(error_msg or "Invalid value", from_typing=from_typing)

    def _clear_error(self) -> None:
        if not self._widget_alive():
            return
        self.component.configure(border_color=self._original_border_color)
        if self._tooltip is not None:
            self._tooltip.clear_error()

    def _validate_and_style(self, value: str, *, from_typing: bool = False) -> bool:
        error = self.validate(value)
        if error is None:
            self._clear_error()
            return True
        self._apply_error(error, from_typing=from_typing)
        return False

    def _on_shadow_write(self, *_args) -> None:
        if self._syncing:
            return
        if not self._touched:
            # external sync or initial set — commit immediately
            self._commit()
            if self._debounce:
                self._debounce.cancel()
            return
        if self._debounce:
            self._debounce.call()
        if self._undo_debounce:
            self._undo_debounce.call()

    def _on_real_var_write(self, *_args) -> None:
        if self._syncing:
            return
        # external change (preset load, file dialog, etc) — sync to shadow var
        self._syncing = True
        self._shadow_var.set(self.var.get())
        self._syncing = False
        self._validate_and_style(self._shadow_var.get())

    def _push_undo_snapshot(self) -> None:
        self._undo.push(self._shadow_var.get())

    def _on_debounce_fire(self) -> None:
        if not self._widget_alive():
            return
        val = self._apply_autocorrect(self._shadow_var.get())
        if self._validate_and_style(val, from_typing=True):
            self._commit()

    def _on_focus_in(self, _e=None) -> None:
        self._touched = False
        self._undo.push(self._shadow_var.get())
        if self._tooltip is not None:
            self._tooltip.on_focus_in()

    def _on_user_input(self, _e=None) -> None:
        self._touched = True
        if self._tooltip is not None:
            self._tooltip.on_user_keystroke()

    def _on_focus_out(self, _e=None) -> None:
        self._cancel_debounces()
        if self._touched:
            val = self._apply_autocorrect(self._shadow_var.get())
            if self._validate_and_style(val):
                self._commit()
        self._undo.push(self._shadow_var.get())
        if self._tooltip is not None:
            self._tooltip.on_focus_out()

    def _on_enter(self, _e=None) -> None:
        self._cancel_debounces()
        if self._touched:
            if self._validate_and_style(self._shadow_var.get()):
                self._commit()

    def _set_value(self, value: str) -> None:
        self._syncing = True
        self._shadow_var.set(value)
        self._syncing = False
        if self._validate_and_style(value):
            self._commit()

    def _on_undo(self, _e=None) -> str:
        previous = self._undo.undo(self._shadow_var.get())
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
        self._model_blank_timer: DebounceTimer | None = None

    def _cancel_debounces(self) -> None:
        super()._cancel_debounces()
        if self._model_blank_timer is not None:
            self._model_blank_timer.cancel()
            self._model_blank_timer = None

    def _get_format_ext(self, default: str = ".safetensors") -> str:
        fmt = self._get_var_value("output_model_format")
        if fmt is None:
            return default
        fmt_str = str(fmt)
        if fmt_str == "DIFFUSERS":
            return ""
        try:
            return ModelFormat[fmt_str].file_extension()
        except KeyError:
            return default

    def _autocorrect_value(self, value: str) -> str:
        if not value:
            return value
        ext = self._get_format_ext("") if self.io_type == PathIOType.MODEL else None
        return autocorrect_path(value, self.io_type, expected_ext=ext)

    def validate(self, value: str) -> str | None:
        base_err = super().validate(value)
        if base_err is not None:
            return base_err
        if value == "":
            return None

        return validate_path(
            value,
            io_type=self.io_type,
            prevent_overwrites=self._get_var_value("prevent_overwrites", False),
            output_format=self._get_var_value("output_model_format"),
        )

    def revalidate(self) -> None:
        if self.component.winfo_exists():
            self._validate_and_style(self._shadow_var.get())

    def _on_debounce_fire(self) -> None:
        super()._on_debounce_fire()
        if self.io_type == PathIOType.MODEL and self._shadow_var.get().strip() == "":
            self._schedule_model_blank_fill()

    def _schedule_model_blank_fill(self) -> None:
        if self._model_blank_timer is not None:
            self._model_blank_timer.cancel()
        self._model_blank_timer = DebounceTimer(
            self.component, DEBOUNCE_TYPING_MS * 2, self._fill_default_model_name,
        )
        self._model_blank_timer.call()

    def _fill_default_model_name(self) -> None:
        if self._shadow_var.get().strip():
            return

        ext = self._get_format_ext()
        use_friendly = bool(self._get_var_value("friendly_run_names", False))

        if use_friendly:
            try:
                name = fw.generate(2, separator="_")
            except Exception:
                name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            method = self._get_var_value("training_method")
            method_str = str(method).lower().replace(" ", "_") if method is not None else "model"
            name = f"{method_str}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self._undo.push(self._shadow_var.get())
        self._set_value(os.path.join("models", f"{name}{ext}"))


def flush_and_validate_all() -> list[str]:
    invalid: list[str] = []

    for v in list(_active_validators):
        if v._debounce:
            v._debounce.cancel()

        value = v._shadow_var.get()
        error = v.validate(value)

        if error is not None:
            v._apply_error(error)
            invalid.append(f"{v.var_name}: {error}")
        else:
            v._clear_error()
            v._commit()

    return invalid
