from __future__ import annotations

import contextlib
import os
import re
import sys
import tkinter as tk
from collections import deque
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from modules.util.config.TrainConfig import get_output_model_destination, is_auto_run_name_mode, prepare_run_name
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.PathIOType import PathIOType
from modules.util.path_util import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from modules.util.ui.autocorrect import (
    INVALID_PATH_CHARS,
    autocorrect_float,
    autocorrect_int,
    autocorrect_path,
    autocorrect_string,
)
from modules.util.ui.ToolTip import ValidationTooltip

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
DEFAULT_MAX_UNDO = 20

_KNOWN_MODEL_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt", ".bin"})
_KNOWN_STRIP_EXTENSIONS = _KNOWN_MODEL_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
_PATH_SEP_RE = re.compile(r"[/\\]")
_INVALID_RUN_NAME_RE = re.compile(r"[^\w .\-]")


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
    if not os.path.exists(abs_path):
        return None

    if is_dir:
        if os.path.isdir(abs_path):
            return f"Output folder `{abs_path}` already exists (overwrite prevented)"
        if os.path.isfile(abs_path):
            return f"Output path `{abs_path}` exists as a file, but a folder is required for this input (overwrite prevented)"
        return f"Output path `{abs_path}` already exists, but is not a folder (overwrite prevented)"

    if os.path.isfile(abs_path):
        return f"Output file `{abs_path}` already exists (overwrite prevented)"
    if os.path.isdir(abs_path):
        return f"Output path `{abs_path}` exists as a folder, but a file is required (overwrite prevented)"
    return f"Output path `{abs_path}` already exists, but is not a file (overwrite prevented)"


def validate_path(
    value: str,
    io_type: PathIOType = PathIOType.INPUT,
    *,
    prevent_overwrites: bool = False,
    output_is_dir: bool = False,
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

    if io_type == PathIOType.OUTPUT:
        if not os.path.isdir(os.path.dirname(os.path.abspath(trimmed))):
            return "Parent folder does not exist"
        if not output_is_dir:
            return _check_overwrite(trimmed, is_dir=False, prevent=prevent_overwrites)

    return None


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
        allow_negative: bool = False,
    ):
        self.component = component
        self.var = var
        self.ui_state = ui_state
        self.var_name = var_name
        self._extra_validate = extra_validate
        self._required = required
        self._allow_negative = allow_negative

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
            return autocorrect_float(value)
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
        if comp._textvariable is not None and comp._textvariable_callback_name:
            with contextlib.suppress(Exception):
                comp._textvariable.trace_remove("write", comp._textvariable_callback_name)
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
                converted = declared_type(value)
                if not self._allow_negative and converted < 0:
                    return "Value must be non-negative"
                if self._required and declared_type is int and converted == 0:
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
        output_is_dir: bool = False,
        max_undo: int = DEFAULT_MAX_UNDO,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(component, var, ui_state, var_name, max_undo=max_undo, extra_validate=extra_validate, required=required)
        self.io_type = io_type
        self.output_is_dir = output_is_dir

    def _autocorrect_value(self, value: str) -> str:
        if not value:
            return value
        return autocorrect_path(value, self.io_type)

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
            output_is_dir=self.output_is_dir,
        )

    def revalidate(self) -> None:
        if self.component.winfo_exists():
            self._validate_and_style(self._shadow_var.get())


class RunNameValidator(FieldValidator):
    """Validator for the run_name field.

    Strips known model/image/video extensions, rejects path separators,
    checks for output-file overwrites, and applies generated names
    when the UI mode or fallback rules require them.
    """

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
        super().__init__(component, var, ui_state, var_name, max_undo=max_undo, extra_validate=extra_validate, required=required)
        self._blank_timer: DebounceTimer | None = None
        self._dep_traces: list[tuple[tk.Variable, str]] = []

    def _cancel_debounces(self) -> None:
        super()._cancel_debounces()
        if self._blank_timer is not None:
            self._blank_timer.cancel()

    def _get_output_model_format(self) -> ModelFormat:
        fmt = self._get_var_value("output_model_format")
        if fmt is None:
            return ModelFormat.SAFETENSORS
        fmt_str = str(fmt)
        try:
            return ModelFormat[fmt_str]
        except KeyError:
            return ModelFormat.SAFETENSORS

    def _autocorrect_value(self, value: str) -> str:
        if not value:
            return value
        result = autocorrect_string(value)
        suffix = PurePosixPath(result).suffix.lower()
        if suffix in _KNOWN_STRIP_EXTENSIONS:
            result = result[: -len(suffix)]
        return result

    def validate(self, value: str) -> str | None:
        base_err = super().validate(value)
        if base_err is not None:
            return base_err
        if not value:
            return None
        if _PATH_SEP_RE.search(value):
            return "Run name must not contain path separators (/ or \\)"

        bad = sorted(set(_INVALID_RUN_NAME_RE.findall(value)))
        if bad:
            shown = ", ".join(_format_char(c) for c in bad[:_MAX_DISPLAY_CHARS])
            if len(bad) > _MAX_DISPLAY_CHARS:
                shown += f" and {len(bad) - _MAX_DISPLAY_CHARS} more"
            return f"Run name contains invalid characters: {shown}"

        suffix = PurePosixPath(value).suffix.lower()
        if suffix in _KNOWN_STRIP_EXTENSIONS:
            return "Run name must not end with a file extension"

        output_dir = str(self._get_var_value("final_output_dir", ""))
        if not output_dir:
            return None
        output_model_format = self._get_output_model_format()
        return _check_overwrite(
            get_output_model_destination(output_dir, value, output_model_format),
            is_dir=not output_model_format.is_single_file(),
            prevent=self._get_var_value("prevent_overwrites", False),
        )

    def _is_auto_mode(self) -> bool:
        return is_auto_run_name_mode(self._get_var_value("run_name_mode", "DEFAULT"))

    def revalidate(self) -> None:
        if self._widget_alive():
            self._validate_and_style(self._shadow_var.get())

    def attach(self) -> None:
        super().attach()
        self._blank_timer = DebounceTimer(
            self.component, 10_000, self._fill_default_run_name,
        )
        for dep_name in ("final_output_dir", "output_model_format", "prevent_overwrites"):
            dep_var = self._get_var_safe(dep_name)
            if dep_var is not None:
                tid = dep_var.trace_add("write", lambda *_a: self.revalidate())
                self._dep_traces.append((dep_var, tid))
        mode_var = self._get_var_safe("run_name_mode")
        if mode_var is not None:
            tid = mode_var.trace_add("write", lambda *_a: self._on_mode_change())
            self._dep_traces.append((mode_var, tid))
        if self._is_auto_mode():
            self._apply_prepared_run_name(is_new_invocation=True)

    def detach(self) -> None:
        for dep_var, tid in self._dep_traces:
            with contextlib.suppress(tk.TclError, ValueError):
                dep_var.trace_remove("write", tid)
        self._dep_traces.clear()
        super().detach()

    def _on_focus_out(self, _e=None) -> None:
        super()._on_focus_out(_e)
        if (self._touched
                and not self._is_auto_mode()
                and self._shadow_var.get().strip() == ""):
            self._apply_prepared_run_name(is_new_invocation=False)

    def _on_debounce_fire(self) -> None:
        super()._on_debounce_fire()
        if (not self._is_auto_mode()
                and self._shadow_var.get().strip() == ""):
            self._schedule_blank_fill()

    def _schedule_blank_fill(self) -> None:
        if self._blank_timer is not None:
            self._blank_timer.call()

    def _on_mode_change(self) -> None:
        if self._is_auto_mode():
            self._apply_prepared_run_name(is_new_invocation=True)
        self.revalidate()

    def _apply_prepared_run_name(self, *, is_new_invocation: bool) -> None:
        current_name = self._shadow_var.get()
        name = prepare_run_name(
            self._get_var_value("training_method", "model"),
            self._get_var_value("run_name_mode", "DEFAULT"),
            current_name,
            is_new_invocation=is_new_invocation,
        )

        self._undo.push(self._shadow_var.get())
        self._set_value(name)

    def _fill_default_run_name(self) -> None:
        if self._shadow_var.get().strip():
            return
        self._apply_prepared_run_name(is_new_invocation=False)


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
