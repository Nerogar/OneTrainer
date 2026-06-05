import platform

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Single stylesheet using palette() references so it works in both light and dark mode
# without hardcoding any colors.
_STYLESHEET = """
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
    # On Linux/macOS, Qt already maps the system light/dark preference onto its
    # palette correctly, so we leave it untouched. Qt's dark mode support on
    # Windows is incomplete (tabs render with no contrast), so we apply explicit
    # fixes there only.
    if platform.system() != "Windows":
        return
    is_dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    if not is_dark:
        # Qt's light palette leaves Base as grey which looks disabled — restore to white.
        palette = app.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
        app.setPalette(palette)
    app.setStyleSheet(_STYLESHEET)
