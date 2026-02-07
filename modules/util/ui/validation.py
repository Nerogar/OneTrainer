from __future__ import annotations

import contextlib
import tkinter as tk
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modules.util.ui.UIState import UIState

    import customtkinter as ctk


DEBOUNCE_TYPING_MS = 250
UNDO_DEBOUNCE_MS = 500
ERROR_BORDER_COLOR = "#dc3545"

_active_validators: set[FieldValidator] = set()

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
    ):
        self.component = component
        self.var = var
        self.ui_state = ui_state
        self.var_name = var_name

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
            if nullable:
                return None
            if declared_type is str:
                if default_val == "":
                    return None
                return "Value required"
            return None

        try:
            if declared_type is int:
                int(value)
            elif declared_type is float:
                float(value)
            elif declared_type is bool:
                if value.lower() not in ("true", "false", "0", "1"):
                    return "Invalid bool"
        except ValueError:
            return "Invalid value"

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
