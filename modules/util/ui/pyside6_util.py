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
    if sys.platform != "win32":
        return

    from ctypes import wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        user32.CopyIcon.argtypes = [wintypes.HICON]
        user32.CopyIcon.restype = wintypes.HICON
        user32.DestroyIcon.argtypes = [wintypes.HICON]
        set_class_icon = user32.SetClassLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user32.SetClassLongW
        set_class_icon.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_class_icon.restype = ctypes.c_size_t
        hwnd = int(window.winId())
        # Replace Qt's generic class icons with copies that live until process exit.
        for icon_type, index in ((1, -14), (0, -34)):
            handle = user32.CopyIcon(user32.SendMessageW(hwnd, 0x7F, icon_type, 0))  # WM_GETICON
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            ctypes.set_last_error(0)
            previous = set_class_icon(hwnd, index, handle)
            if not previous and ctypes.get_last_error():
                error = ctypes.WinError(ctypes.get_last_error())
                user32.DestroyIcon(handle)
                raise error
    except OSError:
        logging.getLogger(__name__).warning("Could not set the Windows taskbar icon", exc_info=True)
