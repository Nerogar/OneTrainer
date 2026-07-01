import signal
import sys
from abc import ABCMeta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


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
    """)

    return app
