import os
import platform
import tkinter as tk
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import unquote, urlparse


def is_dnd_supported_platform() -> bool:
    return platform.system() in ("Windows", "Darwin")


def _normalize_dropped_path(raw_path: str) -> str:
    path = raw_path.strip()

    if path.startswith("{") and path.endswith("}"):
        path = path[1:-1]

    if path.startswith("file://"):
        parsed = urlparse(path)
        path = unquote(parsed.path)

        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]

    return os.path.normpath(path)


def parse_dnd_files(widget: tk.Misc, data: str) -> list[str]:
    """Parse a tkdnd drop payload into normalized filesystem paths."""
    if not data:
        return []

    try:
        raw_paths = widget.tk.splitlist(data)
    except tk.TclError:
        raw_paths = (data,)

    parsed_paths = []
    for raw_path in raw_paths:
        normalized = _normalize_dropped_path(raw_path)
        if normalized:
            parsed_paths.append(normalized)

    return parsed_paths


def _get_drop_target(widget: tk.Misc) -> tk.Misc:
    """Resolves the underlying Tk widget for common CustomTkinter wrappers."""
    for attr in ("_entry", "_textbox", "_parent_canvas", "_canvas", "_text", "_parent_frame"):
        target = getattr(widget, attr, None)
        if target is not None:
            return target
    return widget


def _install_dnd_widget_api(target: tk.Misc, wrapper: type) -> None:
    wrapper_attrs = [
        "_substitute_dnd",
        "_subst_format_dnd",
        "_subst_format_str_dnd",
        "_dnd_bind",
        "dnd_bind",
        "drag_source_register",
        "drag_source_unregister",
        "drop_target_register",
        "drop_target_unregister",
        "platform_independent_types",
        "platform_specific_types",
        "get_dropfile_tempdir",
        "set_dropfile_tempdir",
    ]

    target_class = target.__class__
    for attr_name in wrapper_attrs:
        if hasattr(target_class, attr_name):
            continue
        attr = getattr(wrapper, attr_name, None)
        if attr is not None:
            setattr(target_class, attr_name, attr)


def _ensure_tkdnd_initialized(widget: tk.Misc) -> bool:
    if not is_dnd_supported_platform():
        return False

    try:
        from tkinterdnd2 import TkinterDnD
    except Exception:
        return False

    root = widget.winfo_toplevel()
    if getattr(root, "TkdndVersion", None):
        return True

    require_fn = getattr(TkinterDnD, "_require", None)
    if not callable(require_fn):
        return False

    try:
        root_any = cast(Any, root)
        root_any.TkdndVersion = require_fn(root)

        wrapper = getattr(TkinterDnD, "DnDWrapper", None)
        if wrapper is not None:
            _install_dnd_widget_api(root, wrapper)

        return True
    except Exception:
        return False


def bind_file_drop(widget: tk.Misc, callback: Callable[[list[str]], None]) -> bool:
    """Register a widget as file-drop target and invoke callback with paths."""
    if not is_dnd_supported_platform():
        return False

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except Exception:
        return False

    target = _get_drop_target(widget)
    if not _ensure_tkdnd_initialized(target):
        return False

    wrapper = getattr(TkinterDnD, "DnDWrapper", None)
    if wrapper is not None:
        _install_dnd_widget_api(target, wrapper)

    drop_target_register = getattr(target, "drop_target_register", None)
    dnd_bind = getattr(target, "dnd_bind", None)
    if not callable(drop_target_register) or not callable(dnd_bind):
        return False

    def _on_drop(event: Any):
        paths = parse_dnd_files(target, getattr(event, "data", ""))
        if paths:
            callback(paths)

    try:
        drop_target_register(DND_FILES)
        dnd_bind("<<Drop>>", _on_drop)
        return True
    except Exception:
        return False
