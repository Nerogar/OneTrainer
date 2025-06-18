import logging
import platform
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import EventType
from typing import Any

import customtkinter as ctk
from customtkinter import CTk, CTkToplevel

logger = logging.getLogger(__name__)

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

def _safe_set_widget_value(var_or_entry: Any, value: Any, default_value: Any = None) -> None:
    """
    Helper to safely set the value of a CTk widget (StringVar, BooleanVar, CTkEntry).
    """
    if value is None and default_value is not None:
        value = default_value
    if value is None:  # Still None, skip
        return

    if isinstance(var_or_entry, (ctk.StringVar | ctk.BooleanVar)):
        var_or_entry.set(value)
    elif isinstance(var_or_entry, ctk.CTkEntry):
        var_or_entry.delete(0, "end")
        var_or_entry.insert(0, str(value))

def load_window_session_settings(
    window_instance: Any,
    session_settings_key: str,
    metadata_list: list[tuple[str, str, str, Any]],
) -> dict:
    """
    Loads session settings for a window based on provided metadata.
    Applies defaults if settings are not found or if specific keys are missing.

    Args:
        window_instance: The instance of the CTkToplevel window.
        session_settings_key: The key used to store this window's settings in session_ui_settings.
        metadata_list: A list of tuples, where each tuple defines a setting:
                       (setting_key, target_type, target_name, default_value_or_lambda).
                       target_type can be 'attr' or 'config'.

    Returns:
        The dictionary of saved settings that were found (could be empty if none were found).
    """
    loaded_settings_values = {}
    apply_defaults_to_all = True

    if hasattr(window_instance, "parent") and hasattr(window_instance.parent, "session_ui_settings"):
        saved_settings_for_window = window_instance.parent.session_ui_settings.get(session_settings_key, {})
        if saved_settings_for_window:
            loaded_settings_values = saved_settings_for_window
            apply_defaults_to_all = False # We found settings, so only apply defaults for missing keys
            logger.debug(f"Loaded session settings for key '{session_settings_key}' in {window_instance.__class__.__name__}")
        else:
            logger.debug(f"No saved session settings found for key '{session_settings_key}' in {window_instance.__class__.__name__}. Applying defaults.")
    else:
        logger.warning(f"Parent or session_ui_settings not found for {window_instance.__class__.__name__}. Applying defaults.")

    for setting_key, target_type, target_name, default_val_or_lambda in metadata_list:
        target_widget = None
        if target_type == 'attr':
            if hasattr(window_instance, target_name):
                target_widget = getattr(window_instance, target_name)
        elif target_type == 'config':
            if hasattr(window_instance, "config_state") and isinstance(window_instance.config_state, dict):
                target_widget = window_instance.config_state.get(target_name)

        if target_widget:
            actual_default = default_val_or_lambda
            if callable(default_val_or_lambda):
                try:
                    actual_default = default_val_or_lambda(window_instance)
                except Exception as e:
                    logger.error(f"Error calling default lambda for {target_name} in {window_instance.__class__.__name__}: {e}")
                    actual_default = None

            value_to_set = None
            if not apply_defaults_to_all and setting_key in loaded_settings_values:
                value_to_set = loaded_settings_values.get(setting_key)
                _safe_set_widget_value(target_widget, value_to_set, actual_default)
            else: # Apply default if applying all defaults or key is missing
                _safe_set_widget_value(target_widget, None, actual_default) # Pass None to force default application
        elif target_type in ['attr', 'config']:
             logger.warning(f"Target widget for setting '{setting_key}' (type: {target_type}, name: {target_name}) not found in {window_instance.__class__.__name__}.")

    return loaded_settings_values

def save_window_session_settings(
    window_instance: Any,
    session_settings_key: str,
    metadata_list: list[tuple[str, str, str, Any]],
    additional_settings: dict | None = None
) -> None:
    """
    Saves session settings for a window based on provided metadata.

    Args:
        window_instance: The instance of the CTkToplevel window.
        session_settings_key: The key used to store this window's settings in session_ui_settings.
        metadata_list: A list of tuples, where each tuple defines a setting:
                       (setting_key, target_type, target_name, default_value_or_lambda - not used for saving).
        additional_settings: An optional dictionary of settings to merge, for special cases.
    """
    if not hasattr(window_instance, "parent") or not hasattr(window_instance.parent, "session_ui_settings"):
        logger.warning(f"Parent or session_ui_settings not found for {window_instance.__class__.__name__} during save.")
        return

    current_settings_to_save = {}
    for setting_key, target_type, target_name, _ in metadata_list:
        value_to_save = None
        target_widget = None

        if target_type == 'attr':
            if hasattr(window_instance, target_name):
                target_widget = getattr(window_instance, target_name)
        elif target_type == 'config':
            if hasattr(window_instance, "config_state") and isinstance(window_instance.config_state, dict):
                target_widget = window_instance.config_state.get(target_name)

        if target_widget:
            if isinstance(target_widget, (ctk.StringVar | ctk.BooleanVar | ctk.CTkEntry)):
                value_to_save = target_widget.get()
            else:
                logger.warning(f"Unsupported widget type for saving: {type(target_widget)} for setting '{setting_key}' in {window_instance.__class__.__name__}")
        current_settings_to_save[setting_key] = value_to_save

    if additional_settings:
        current_settings_to_save.update(additional_settings)

    window_instance.parent.session_ui_settings[session_settings_key] = current_settings_to_save
    logger.debug(f"Saved session settings for key '{session_settings_key}' in {window_instance.__class__.__name__}")
