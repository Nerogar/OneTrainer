from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from modules.util.enum.PathIOType import PathIOType
from modules.util.ui.validation import (
    DEBOUNCE_TYPING_MS,
    DEFAULT_MAX_UNDO,
    ERROR_BORDER_COLOR,
    UNDO_DEBOUNCE_MS,
    BaseFieldValidator,
    UndoHistory,
    _active_validators,
    _validate_path_field,
)

if TYPE_CHECKING:
    from modules.util.ui.CtkUIState import CtkUIState

    import customtkinter as ctk


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


class FieldValidator(BaseFieldValidator):
    def __init__(
        self,
        component: ctk.CTkEntry,
        var: tk.Variable,
        ui_state: CtkUIState,
        var_name: str,
        max_undo: int = DEFAULT_MAX_UNDO,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(ui_state, var_name, extra_validate, required)
        self.component = component
        self.var = var

        try:
            self._original_border_color = component.cget("border_color")
        except Exception:
            self._original_border_color = "gray50"

        self._shadow_var = tk.StringVar(master=component)
        self._shadow_trace_name: str | None = None
        self._real_var_trace_name: str | None = None
        self._syncing = False
        self._touched = False

        self._debounce: DebounceTimer | None = None
        self._undo_debounce: DebounceTimer | None = None
        self._undo = UndoHistory(max_undo)

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

        if self._debounce:
            self._debounce.cancel()
        if self._undo_debounce:
            self._undo_debounce.cancel()

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
            comp._textvariable_callback_name = new_var.trace_add(
                "write", comp._textvariable_callback
            )

    def _commit(self) -> None:
        shadow_val = self._shadow_var.get()
        if shadow_val != self.var.get():
            self._syncing = True
            self.var.set(shadow_val)
            self._syncing = False

    def _apply_error(self) -> None:
        self.component.configure(border_color=ERROR_BORDER_COLOR)

    def _clear_error(self) -> None:
        self.component.configure(border_color=self._original_border_color)

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
        val = self._shadow_var.get()
        if self._validate_and_style(val):
            self._commit()

    def _on_focus_in(self, _e=None) -> None:
        self._touched = False
        self._undo.push(self._shadow_var.get())

    def _on_user_input(self, _e=None) -> None:
        self._touched = True

    def _on_focus_out(self, _e=None) -> None:
        if self._debounce:
            self._debounce.cancel()
        if self._undo_debounce:
            self._undo_debounce.cancel()
        if self._touched:
            if self._validate_and_style(self._shadow_var.get()):
                self._commit()
        self._undo.push(self._shadow_var.get())

    def _on_enter(self, _e=None) -> None:
        if self._debounce:
            self._debounce.cancel()
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

    def flush(self) -> str | None:
        if self._debounce:
            self._debounce.cancel()

        value = self._shadow_var.get()
        error = self.validate(value)

        if error is not None:
            self._apply_error()
        else:
            self._clear_error()
            self._commit()

        return error


class PathValidator(FieldValidator):
    """FieldValidator with additional path-specific checks."""

    def __init__(
        self,
        component: ctk.CTkEntry,
        var: tk.Variable,
        ui_state: CtkUIState,
        var_name: str,
        io_type: PathIOType = PathIOType.INPUT,
        max_undo: int = DEFAULT_MAX_UNDO,
        extra_validate: Callable[[str], str | None] | None = None,
        required: bool = False,
    ):
        super().__init__(component, var, ui_state, var_name, max_undo=max_undo, extra_validate=extra_validate, required=required)
        self.io_type = io_type

    def validate(self, value: str) -> str | None:
        base_err = super().validate(value)
        if base_err is not None:
            return base_err
        if value == "":
            return None
        return _validate_path_field(self.ui_state, self.io_type, value)

    def revalidate(self) -> None:
        if self.component.winfo_exists():
            self._validate_and_style(self._shadow_var.get())
