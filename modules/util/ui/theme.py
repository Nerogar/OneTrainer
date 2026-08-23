import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Lets reviewers whose OS never reports dark still see the dark theme, by pretending it did.
OT_FORCE_DARK = os.environ.get("OT_FORCE_DARK") == "1"

_BASE_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        padding: 2px 2px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
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
"""

_LIGHT_STYLESHEET = """
    QProgressBar {
        background-color: #c8c8c8;
    }
"""

def apply_theme(app: QApplication) -> None:
    if OT_FORCE_DARK:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
        is_dark = True
    else:
        is_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark

    palette = app.palette()
    if not is_dark:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)
    app.setStyleSheet(_BASE_STYLESHEET if is_dark else _BASE_STYLESHEET + _LIGHT_STYLESHEET)
