import html
import locale
import os
import re
import signal
import sys
from abc import ABCMeta
from typing import TYPE_CHECKING

from modules.util.ui import pyside6_components as components
from modules.util.ui.theme import ThemeController

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QStyleFactory, QWidget

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


def limit_input_width(entry: QLineEdit, characters: int = 40) -> None:
    """Cap the visible width without limiting the length of the entered text."""
    entry.ensurePolished()
    entry.setMaximumWidth(entry.fontMetrics().horizontalAdvance("0" * characters) + 12)


def fit_form(frame: QWidget) -> None:
    """Fit dropdowns to their options and put surplus space after the form."""
    for combo in frame.findChildren(QComboBox):
        combo.ensurePolished()
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        combo.setFixedWidth(combo.sizeHint().width())
    components._pack_form(frame)
def format_tooltip(text: str) -> str:
    # wrap tooltips after 45 characters, supported by studys size
    wrapped = re.sub(r"(.{65}\S*)[ \t]+", r"\1\n", text)
    return f'<p style="white-space: pre;">{html.escape(wrapped)}</p>'


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
