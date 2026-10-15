import locale
import signal
import sys
from abc import ABCMeta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


def create_application() -> QApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
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
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)

    app.setStyleSheet("""
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
            padding: 2px 2px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QProgressBar {
            background-color: #c8c8c8;
        }
        QToolButton {
            padding-top: 0px;
            padding-bottom: 0px;
            padding-right: 40px;
        }
        QToolButton::menu-indicator {
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 12px;
            height: 12px;
            right: 10px;
        }
    """)

    return app
