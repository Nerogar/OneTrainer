import contextlib
import os
import platform
import sys
import tkinter as tk
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from tkinter import EventType
from typing import Any

from customtkinter import CTk, CTkToplevel
from tkinterdnd2 import DND_FILES


def bind_mousewheel(
    widget: Any,
    whitelist: set[Any] | None,
    callback: Callable[[int, Any], None],
):
    assert whitelist is None or isinstance(whitelist, set)

    is_mac = sys.platform == "darwin"

    def process_mousewheel(raw_event):
        # If whitelist was provided, only respond to events on allowed widgets.
        if whitelist is not None and raw_event.widget not in whitelist:
            return

        # Cross-platform mouse scroll handler.
        # SEE: Section "54.6" of https://tkdocs.com/shipman/tkinter.pdf, which
        # describes the `.delta` and `.num` property behaviors on each platform.
        if raw_event.type == EventType.MouseWheel:  # Windows and Mac.
            # Positive sign means scroll up, negative sign means scroll down.
            # Windows uses a multiple of 120. Macs use the raw number of steps.
            delta = (
                raw_event.delta if is_mac else int(raw_event.delta / 120)
            )
        elif raw_event.type == EventType.ButtonPress:  # Linux.
            # Button 4 means scroll up. Button 5 means scroll down.
            # NOTE: Tk only supports binding mouse buttons 1, 2 and 3. The 4/5
            # values are ONLY used for indicating mousewheel scrolling.
            delta = 1 if raw_event.num == 4 else -1
        else:
            raise Exception(f"unhandled event type: {raw_event.type.name}")

        # We provide the raw event too, if they want to analyze it further.
        callback(delta, raw_event)

    widget.bind("<MouseWheel>", process_mousewheel)
    if sys.platform == "linux":
        widget.bind("<Button-4>", process_mousewheel)
        widget.bind("<Button-5>", process_mousewheel)

def set_window_icon(window: tk.Tk | tk.Toplevel | CTk | CTkToplevel) -> None:
    """Set the application window icon based on the current platform

    Args:
        window: The window object (Tk, Toplevel, CTk, CTkToplevel) to set the icon for
    """
    # Early exit if not a valid window
    if not hasattr(window, "wm_title"):
        return

    # Get icon paths based on platform
    icon_dir = Path("resources/icons")
    system = platform.system()

    try:
        # Check if it's a root window or toplevel window
        is_root_window = isinstance(window, (tk.Tk | CTk))

        if system == "Windows":
            # Windows - use .ico file
            ico_path = icon_dir / "icon.ico"
            if ico_path.exists():
                window.wm_iconbitmap(str(ico_path))

        elif system == "Linux":
            # Linux - use .png with PhotoImage
            png_path = icon_dir / "icon.png"
            if png_path.exists():
                if is_root_window:
                    # For root windows - set immediately
                    window._icon_image_ref = tk.PhotoImage(file=str(png_path))
                    window.iconphoto(False, window._icon_image_ref)
                else:
                    # For toplevels - use delayed setting
                    window.wm_iconbitmap() # Clear any existing icon

                    def set_icon():
                        try:
                            window._icon_image_ref = tk.PhotoImage(file=str(png_path))
                            window.iconphoto(False, window._icon_image_ref)
                        except Exception as e:
                            print(f"Failed to set Linux window icon: {e}")

                    window.after(100, set_icon) # Delay on linux as found less reliable

        elif system == "Darwin":  # macOS
            # macOS uses app bundles for icons, Tkinter support is limited
            pass

    except Exception as e:
        print(f"Failed to set window icon: {e}")

def _drop_enter(event):
    event.widget.focus_force()
    return event.action

def _drop_leave(event):
    return event.action

def _create_drop_handler(entry_widget, ui_state, var_name, command=None, drop_validator=None, on_reject=None):
    def drop(event):
        if event.data:
            paths = _parse_dropped_paths(event.data)
            if paths:
                dropped_path = paths[0]

                if drop_validator and not drop_validator(dropped_path):
                    if on_reject:
                        on_reject(dropped_path)
                    return event.action

                ui_state.get_var(var_name).set(dropped_path)

                entry_widget.focus_force()
                entry_widget.event_generate('<FocusIn>')
                entry_widget.event_generate('<Key>')
                entry_widget.event_generate('<FocusOut>')

                if command:
                    command(dropped_path)
        return event.action
    return drop

def _parse_dropped_paths(event_data: str) -> list[str]:
    paths = []
    current_path = ""
    in_braces = False

    for char in event_data:
        if char == '{':
            in_braces = True
        elif char == '}':
            in_braces = False
        elif char == ' ' and not in_braces:
            pass  # Will append below
        else:
            current_path += char
            continue

        if current_path:
            paths.append(current_path)
            current_path = ""

    if current_path:
        paths.append(current_path)


    cleaned_paths = []
    for p in paths:
        p = p.strip()
        if not p:
            continue

        if p.startswith('file://'):
            p = urllib.parse.unquote(p[7:])  # remove 'file://' and decode %xx
            if p.startswith('//'):
                p = p[2:]  # file:///path -> /path

        cleaned_paths.append(p)

    return cleaned_paths


def register_drop_target(entry_widget, ui_state, var_name, command=None, drop_validator=None, on_reject=None):
    # tkinterdnd2 is unstable on Linux https://github.com/Eliav2/tkinterdnd2/issues/12#issuecomment-3776598066
    if sys.platform == "linux":
        return

    try:
        entry_widget.drop_target_register(DND_FILES)
        for event, handler in [('<<DropEnter>>', _drop_enter), ('<<DropLeave>>', _drop_leave),
                               ('<<Drop>>', _create_drop_handler(entry_widget, ui_state, var_name, command, drop_validator, on_reject))]:
            entry_widget.dnd_bind(event, handler)
    except Exception as e:
        print(f"Failed to register drop target: {e}")

def register_concept_drop_target(widget, drop_callback: Callable[[str], None], allow_multiple: bool = True):
    if sys.platform == "linux":
        return

    def drop_handler(event):
        if not event.data:
            return event.action

        paths = _parse_dropped_paths(event.data)

        for path in paths:
            path = os.path.dirname(path) if os.path.isfile(path) else path
            if os.path.isdir(path):
                drop_callback(path)
                if not allow_multiple:
                    break

        return event.action

    try:
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind('<<DropEnter>>', _drop_enter)
        widget.dnd_bind('<<DropLeave>>', _drop_leave)
        widget.dnd_bind('<<Drop>>', drop_handler)
    except Exception as e:
        print(f"Failed to register concept drop target: {e}")

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
        """Cancel any pending debounced callback."""
        if self._after_id:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)
            self._after_id = None

    def flush(self):
        """Execute any pending callback immediately and cancel the scheduled timer."""
        if self._after_id:
            with contextlib.suppress(tk.TclError):
                self.widget.after_cancel(self._after_id)
            after_id = self._after_id
            self._after_id = None
            # Only execute if pending callback
            if after_id:
                self.callback()
