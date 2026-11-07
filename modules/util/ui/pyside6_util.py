import ctypes
import locale
import logging
import os
import signal
import sys
from abc import ABCMeta
from typing import TYPE_CHECKING

from modules.util.ui.theme import ThemeController

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

if TYPE_CHECKING:
    # pyside6 stubs omit qts runtime metaclass; expose ABCMeta to type checkers.
    QtABCMeta = ABCMeta
else:
    # reachable at runtime
    class QtABCMeta(type(QWidget), ABCMeta):
        # Combine Qt Shiboken metaclass with ABCMeta at runtime.
        pass


class OneTrainerApplication(QApplication):
    theme_controller: ThemeController


def get_theme_controller() -> ThemeController:
    app = QApplication.instance()
    if not isinstance(app, OneTrainerApplication):
        raise RuntimeError("The OneTrainer application has not been created")
    return app.theme_controller


def create_application() -> OneTrainerApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if sys.platform == "win32":
        try:
            ctypes.OleDLL("shell32").SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p("OneTrainer.OneTrainer"))
        except OSError:
            logging.getLogger(__name__).warning("Could not set the Windows taskbar identity", exc_info=True)

    app = OneTrainerApplication(sys.argv)
    app.setOrganizationName("OneTrainer")
    app.setApplicationName("OneTrainer")
    # On desktops without the xdg-desktop-portal Settings interface, Qt spams two
    # "qt.qpa.theme.gnome: dbus reply error ... org.freedesktop.portal.Settings"
    # lines while probing for the system color scheme. Silence just that category;
    # the rules string is read when Qt's logging initializes at QApplication init.
    _gnome_theme_rule = "qt.qpa.theme.gnome=false"
    existing_rules = os.environ.get("QT_LOGGING_RULES")
    os.environ["QT_LOGGING_RULES"] = f"{existing_rules};{_gnome_theme_rule}" if existing_rules else _gnome_theme_rule

    # QApplication initializes the C locale from the environment (setlocale(LC_ALL, "")), which sets LC_NUMERIC
    # to a locale whose decimal separator may be a comma. C libraries then misparse '.' floats: protobuf's upb
    # backend rejects sentencepiece's schema ("Invalid default '0.9995'"), breaking every sentencepiece tokenizer
    # (T5/Chroma/Flux), and the Kineto profiler writes comma decimals into its JSON traces. Restore the C numeric
    # locale, which Python itself uses by default. Qt's own display uses QLocale, which is independent of this and
    # keeps formatting numbers per the system locale.
    locale.setlocale(locale.LC_NUMERIC, "C")
    # Force Fusion everywhere: native styles (e.g. windowsvista) draw standard
    # controls via OS theme APIs, which breaks once an application stylesheet
    # is set, producing a flatter look than Fusion's own stylesheet-aware painting.
    app.setStyle(QStyleFactory.create("Fusion"))

    app.theme_controller = ThemeController(app, QSettings())

    return app


def set_windows_taskbar_icon(window: QWidget) -> None:
    """Copy Qt's window icons into the Windows class icons used by the taskbar."""
    if sys.platform != "win32":
        return

    from ctypes import wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Tell Python the input and output types of these Windows functions.
        # Handles are Windows object IDs; the types must preserve all 64 bits on 64-bit Python.
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        user32.CopyIcon.argtypes = [wintypes.HICON]
        user32.CopyIcon.restype = wintypes.HICON
        user32.DestroyIcon.argtypes = [wintypes.HICON]
        set_class_icon = user32.SetClassLongPtrW
        set_class_icon.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_class_icon.restype = ctypes.c_size_t
        window_handle = int(window.winId())
        wm_geticon = 0x007F  # Windows message asking a window for its icon.

        # Keep independent copies alive until process exit, even if Qt replaces its icons.
        # This workaround was AI-driven; no solution was found on the Qt forum and windows ctypes are not my domain but we needed a icon fix
        for icon_type, class_icon_slot in (
            (1, -14),  # ICON_BIG -> GCLP_HICON (large class icon)
            (0, -34),  # ICON_SMALL -> GCLP_HICONSM (small class icon)
        ):
            qt_icon = user32.SendMessageW(window_handle, wm_geticon, icon_type, 0)
            icon_copy = user32.CopyIcon(qt_icon)
            if not icon_copy:
                raise ctypes.WinError(ctypes.get_last_error())
            # A zero return can mean success, so also check Windows' error code.
            ctypes.set_last_error(0)
            previous = set_class_icon(window_handle, class_icon_slot, icon_copy)
            if not previous and ctypes.get_last_error():
                error = ctypes.WinError(ctypes.get_last_error())
                user32.DestroyIcon(icon_copy)
                raise error
    except OSError:
        logging.getLogger(__name__).warning("Could not set the Windows taskbar icon", exc_info=True)
