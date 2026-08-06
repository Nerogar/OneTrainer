import platform

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

IS_WINDOWS = platform.system() == "Windows"

_BASE_STYLESHEET = """
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
"""

def apply_theme(app: QApplication) -> None:
    is_dark =  app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    palette = app.palette()
    if not IS_WINDOWS or not is_dark:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        palette = app.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)
    app.setStyleSheet(_BASE_STYLESHEET)
