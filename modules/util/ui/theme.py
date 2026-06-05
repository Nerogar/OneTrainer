import platform

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Light mode keeps the original stylesheet unchanged.
_LIGHT_STYLESHEET = """
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

# Dark mode adds tab styling (Qt renders tabs with no contrast on Windows) and uses
# palette() references so colors follow the system theme.
_DARK_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        padding: 2px 2px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
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


def apply_theme(app: QApplication) -> None:
    # Qt's dark mode is broken on Windows (tabs render with no contrast); fix it there
    # only — Linux/macOS map the system theme onto the palette correctly already.
    if platform.system() != "Windows":
        return
    is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    if is_dark:
        app.setStyleSheet(_DARK_STYLESHEET)
    else:
        palette = app.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
        app.setPalette(palette)
        app.setStyleSheet(_LIGHT_STYLESHEET)
