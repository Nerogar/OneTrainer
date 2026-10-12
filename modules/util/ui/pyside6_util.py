import locale
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

    app = OneTrainerApplication(sys.argv)
    app.setOrganizationName("OneTrainer")
    app.setApplicationName("OneTrainer")
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
