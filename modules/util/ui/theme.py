import platform

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

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
"""

_WINDOWS_OVERRIDES = """
    QProgressBar {
        background-color: palette(mid);
    }
    QTabWidget::pane {
        border: 1px solid palette(mid);
    }
    QTabBar::tab {
        background-color: palette(window);
        color: palette(window-text);
        border: 1px solid palette(mid);
        border-bottom: none;
        padding: 4px 8px;
    }
    QTabBar::tab:selected {
        background-color: palette(highlight);
        color: palette(highlighted-text);
    }
    QTabBar::tab:!selected:hover {
        background-color: palette(button);
    }
"""


def _apply_light_base(app: QApplication) -> None:
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)


def apply_theme(app: QApplication) -> None:
    if platform.system() != "Windows":
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        _apply_light_base(app)
        app.setStyleSheet(_BASE_STYLESHEET)
        return

    is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    if not is_dark:
        _apply_light_base(app)
    app.setStyleSheet(_BASE_STYLESHEET + _WINDOWS_OVERRIDES)
