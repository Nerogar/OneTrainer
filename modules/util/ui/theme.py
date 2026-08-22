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

# A scheme change regenerates only the palette roles nobody set explicitly. Base is set below
# for light mode, so it would keep that white through every later switch; each scheme is applied
# from a pristine copy taken before the first override instead.
_scheme_palettes = {}

# Whether the dark palette is the one currently applied.
is_dark_theme = False


def _capture_scheme_palettes(app: QApplication) -> None:
    original_scheme = app.styleHints().colorScheme()
    for scheme in (Qt.ColorScheme.Light, Qt.ColorScheme.Dark):
        app.styleHints().setColorScheme(scheme)
        _scheme_palettes[scheme] = QPalette(app.palette())
    app.styleHints().setColorScheme(original_scheme)


def apply_theme(app: QApplication, dark: bool | None = None) -> None:
    global is_dark_theme

    if not _scheme_palettes:
        _capture_scheme_palettes(app)

    if dark is None:
        is_dark =  app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        dark = IS_WINDOWS and is_dark
    is_dark_theme = dark

    scheme = Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
    app.styleHints().setColorScheme(scheme)
    palette = QPalette(_scheme_palettes[scheme])
    if not dark:
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)
    app.setStyleSheet(_BASE_STYLESHEET)
