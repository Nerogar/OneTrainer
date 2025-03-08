import platform
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import EventType
from typing import Any


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


def set_window_icon(window: Any) -> None:
    """Set the application window icon based on the current platform

    Args:
        window: The window object (Tk, Toplevel, CTk, CTkToplevel) to set the icon for
    """
    # Check if window is a valid window object that can have an icon
    if (
        not hasattr(window, "winfo_toplevel")
        and not hasattr(window, "iconbitmap")
        and not hasattr(window, "iconphoto")
    ):
        # Not a window that can have icons
        return
    icon_dir = (
        Path(__file__).parent.parent.parent.parent / "resources/icons"
    )

    try:
        if platform.system() == "Windows":
            ico_path = icon_dir / "icon.ico"
            if ico_path.exists():
                # For windows, use the toplevel window if this is a frame
                if hasattr(window, "winfo_toplevel") and not hasattr(
                    window, "iconbitmap"
                ):
                    window = window.winfo_toplevel()
                window.iconbitmap(str(ico_path))
        elif platform.system() == "Linux":
            png_path = icon_dir / "icon.png"
            if png_path.exists():
                icon_img = tk.PhotoImage(file=str(png_path))
                # For Linux, use the toplevel window if this is a frame
                if hasattr(window, "winfo_toplevel") and not hasattr(
                    window, "iconphoto"
                ):
                    window = window.winfo_toplevel()
                window.iconphoto(True, icon_img)
        elif platform.system() == "Darwin":  # macOS
            # macOS are a rabbit hole sadly.
            pass
    except Exception as e:
        print(f"Failed to set window icon: {e}")
