import contextlib
import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

_WAIT_TIME_MS = 500
_WRAP_LENGTH = 180
_WRAP_LENGTH_WIDE = 350


class InfoTooltip:
    """Create a hover tooltip for a given widget."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str = "widget info",
        y_offset: int = 20,
        *,
        wide_wrap: bool = False,
    ):
        self.widget = widget
        self.text = text
        self._y_offset = y_offset

        self._wait_time = _WAIT_TIME_MS  # milliseconds
        self._wrap_length = _WRAP_LENGTH_WIDE if wide_wrap else _WRAP_LENGTH
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)
        self.widget.bind("<ButtonPress>", self._on_leave)
        self._after_id: str | None = None
        self._toplevel: ctk.CTkToplevel | None = None

    def _on_enter(self, _event: tk.Event | None = None):
        self._schedule()

    def _on_leave(self, _event: tk.Event | None = None):
        self._unschedule()
        self._hide_tip()

    def _schedule(self):
        self._unschedule()
        self._after_id = self.widget.after(self._wait_time, self._show_tip)

    def _unschedule(self):
        after_id = self._after_id
        self._after_id = None
        if after_id:
            self.widget.after_cancel(after_id)

    def _show_tip(self, _event: tk.Event | None = None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + self._y_offset
        # Creates a toplevel window
        self._toplevel = ctk.CTkToplevel(self.widget)
        # Leaves only the label and removes the app window
        self._toplevel.wm_overrideredirect(True)
        self._toplevel.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(
            self._toplevel, text=self.text, justify="left",
            wraplength=self._wrap_length,
        )
        label.pack(padx=8, pady=8)

    def _hide_tip(self):
        toplevel = self._toplevel
        self._toplevel = None
        if toplevel:
            toplevel.destroy()


_TOOLTIP_AUTO_HIDE_MS = 5000
_TOOLTIP_BG_COLOR = "#FFECEC"
_TOOLTIP_TEXT_COLOR = "#5A0D0D"
_VALIDATION_WRAP_LENGTH = 300


class ValidationTooltip:
    """Error tooltip, only one visible at a time, auto-hides after 5 seconds."""

    # Class-level singleton tracking
    _active_instance: "ValidationTooltip | None" = None

    def __init__(self, widget: tk.Widget, text: str = ""):
        self.widget = widget
        self.text = text

        self._toplevel: ctk.CTkToplevel | None = None
        self._label: ctk.CTkLabel | None = None
        self._configure_binds: list[tuple[tk.Misc, str, str]] = []
        self._auto_hide_id: str | None = None

        self._hovered: bool = False
        self._focused: bool = False
        self._suppressed_until_keystroke: bool = False
        self._error_active: bool = False

        self._bind_ids: list[tuple[str, str]] = []
        self._bind("<Enter>", self._on_enter)
        self._bind("<Leave>", self._on_leave)
        self._bind("<Unmap>", lambda e: self._hide())

    def _bind(self, sequence: str, handler: Callable[[tk.Event], None]):
        with contextlib.suppress(tk.TclError):
            func_id = self.widget.bind(sequence, handler, add=True)
            self._bind_ids.append((sequence, func_id))

    def show_error(self, text: str, *, from_typing: bool = False):
        self.text = text
        self._error_active = True

        if from_typing:
            self._suppressed_until_keystroke = False
            self._show()
            self._schedule_auto_hide()
        elif self._hovered or (self._focused and not self._suppressed_until_keystroke):
            self._show()

    def clear_error(self):
        self._error_active = False
        self._hide()

    def on_user_keystroke(self):
        self._suppressed_until_keystroke = False
        if self._error_active and self._focused:
            self._show()
            self._schedule_auto_hide()

    def on_focus_in(self):
        self._focused = True
        if self._error_active and not self._suppressed_until_keystroke:
            self._show()

    def on_focus_out(self):
        self._focused = False
        if not self._hovered:
            self._hide()

    def _on_enter(self, _e=None):
        self._hovered = True
        if self._error_active:
            self._show()

    def _on_leave(self, _e=None):
        self._hovered = False
        if not self._focused:
            self._hide()

    def _show(self):
        if ValidationTooltip._active_instance is not None and ValidationTooltip._active_instance is not self:
            ValidationTooltip._active_instance._hide()
        ValidationTooltip._active_instance = self

        try:
            root = self.widget.winfo_toplevel()
            state = root.wm_state()
            if state in ("iconic", "withdrawn"):
                return
        except tk.TclError:
            return

        if self._toplevel is not None:
            self._update_text()
            self._reposition()
            return

        try:
            if not self.widget.winfo_exists():
                return
        except tk.TclError:
            return

        x, y = self._calc_position()
        toplevel = ctk.CTkToplevel(self.widget)
        toplevel.wm_overrideredirect(True)
        toplevel.wm_geometry(f"+{x}+{y}")
        toplevel.attributes("-topmost", True)
        toplevel.configure(fg_color=_TOOLTIP_BG_COLOR)

        self._label = ctk.CTkLabel(
            toplevel, text=self.text, justify="left",
            wraplength=_VALIDATION_WRAP_LENGTH,
            text_color=_TOOLTIP_TEXT_COLOR,
            fg_color=_TOOLTIP_BG_COLOR,
        )
        self._label.pack(padx=8, pady=6)

        self._toplevel = toplevel

        self._bind_configure_events()

    def _hide(self):
        self._cancel_auto_hide()
        self._unbind_configure_events()

        if ValidationTooltip._active_instance is self:
            ValidationTooltip._active_instance = None

        toplevel = self._toplevel
        self._toplevel = None
        if toplevel is not None:
            with contextlib.suppress(tk.TclError):
                toplevel.destroy()

    def _update_text(self):
        if self._toplevel is not None and self._label is not None:
            with contextlib.suppress(tk.TclError):
                self._label.configure(text=self.text)

    def _calc_position(self) -> tuple[int, int]:
        try:
            x = self.widget.winfo_rootx()
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            x, y = 0, 0
        return x, y

    def _reposition(self):
        if self._toplevel is None:
            return
        x, y = self._calc_position()
        with contextlib.suppress(tk.TclError):
            self._toplevel.wm_geometry(f"+{x}+{y}")

    def _bind_configure_events(self):
        self._unbind_configure_events()

        targets: list[tk.Misc] = [self.widget]
        with contextlib.suppress(tk.TclError):
            targets.append(self.widget.winfo_toplevel())

        def _on_configure(e=None):
            if self._toplevel is not None and getattr(e, "widget", None) in targets:
                self._reposition()

        for target in targets:
            with contextlib.suppress(tk.TclError):
                func_id = target.bind("<Configure>", _on_configure, add=True)
                self._configure_binds.append((target, "<Configure>", func_id))

    def _unbind_configure_events(self):
        for target, seq, func_id in self._configure_binds:
            with contextlib.suppress(tk.TclError):
                target.unbind(seq, func_id)
        self._configure_binds.clear()

    def _schedule_auto_hide(self):
        self._cancel_auto_hide()

        def _fire():
            self._auto_hide_id = None
            self._suppressed_until_keystroke = True
            if not self._hovered:
                self._hide()

        try:
            self._auto_hide_id = self.widget.after(_TOOLTIP_AUTO_HIDE_MS, _fire)
        except tk.TclError:
            self._auto_hide_id = None

    def _cancel_auto_hide(self):
        if self._auto_hide_id is not None:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._auto_hide_id)
            self._auto_hide_id = None

    def destroy(self):
        self._hide()
        for seq, fid in self._bind_ids:
            with contextlib.suppress(tk.TclError):
                self.widget.unbind(seq, fid)
        self._bind_ids.clear()
