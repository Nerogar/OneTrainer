import os

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Lets reviewers whose OS never reports dark still see the dark theme, by pretending it did.
OT_FORCE_DARK = os.environ.get("OT_FORCE_DARK") == "1"

_BASE_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        padding: 2px 2px;
        border-radius: 2px;
    }
    QPushButton[listAction="remove"], QPushButton[listAction="clone"] {
        border: 1px solid transparent;
        border-radius: 2px;
        padding: 0;
        font-size: 18px;
    }
    QPushButton[listAction="remove"] {
        background-color: #FFECEC;
        color: #5A0D0D;
        border-color: #5A0D0D;
    }
    QPushButton[listAction="clone"] {
        background-color: #D7F5EA;
        color: #0A4933;
        border-color: #0A4933;
        font-weight: bold;
    }
    QPushButton[listAction="remove"]:focus, QPushButton[listAction="clone"]:focus {
        border-style: dotted;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    QCheckBox[largeIndicator="true"]::indicator {
        width: 20px;
        height: 20px;
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
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        background-color: white;
        border: 1px solid #c8c8c8;
    }
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
        background-color: #e0e0e0;
    }
    QProgressBar {
        background-color: #c8c8c8;
    }
"""

_DARK_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
        background-color: #404040;
        color: palette(text);
        border: 1px solid #242424;
    }
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
        background-color: #282828;
        color: #808080;
    }
    /* Include the 1px border in the normal 16px / large 20px indicator size. */
    QCheckBox::indicator:unchecked {
        width: 14px;
        height: 14px;
        background-color: palette(window);
        border: 1px solid palette(midlight);
        border-radius: 2px;
    }
    QCheckBox[largeIndicator="true"]::indicator:unchecked {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:unchecked:enabled:hover,
    QCheckBox::indicator:unchecked:enabled:focus {
        border-color: palette(text);
    }
    QCheckBox::indicator:unchecked:enabled:pressed {
        background-color: palette(mid);
    }
    QCheckBox::indicator:unchecked:disabled {
        border-color: palette(mid);
    }
"""


def _form_stylesheet(palette: QPalette) -> str:
    def blend(background: QColor, foreground: QColor, amount: float) -> QColor:
        return QColor(*(round(a + (b - a) * amount) for a, b in zip(
            background.getRgb()[:3], foreground.getRgb()[:3], strict=True,
        )))

    base = palette.color(QPalette.ColorRole.Window)
    text = palette.color(QPalette.ColorRole.WindowText)
    dark = base.lightnessF() < 0.5
    surface = blend(base, QColor("white"), 0.065 if dark else 0.65)
    border = blend(base, QColor("white" if dark else "black"), 0.13 if dark else 0.12)
    dark_inset = surface.lightnessF() < 0.5
    inset = blend(surface, QColor("black"), 0.12 if dark_inset else 0.025)
    fill = blend(inset, text, 0.12 if dark_inset else 0.06)
    disabled_fill = blend(inset, text, 0.04)
    disabled_text = blend(disabled_fill, text, 0.38)
    return f"""
        QLabel[formLabel="heading"]:enabled {{ color: {text.name()}; }}
        QLabel[formLabel="field"]:enabled {{ color: {blend(text, base, 0.12).name()}; }}
        QFrame[formPanel="true"] {{
            background-color: {surface.name()};
            border: 1px solid {border.name()};
            border-radius: 6px;
        }}
        QFrame[formInset="true"] {{
            background-color: {inset.name()};
            border: 1px solid {blend(inset, text, 0.20).name()};
            border-radius: 6px;
        }}
        QFrame[formInset="true"] QLineEdit {{
            background-color: {fill.name()};
            color: {text.name()};
            border: none;
            border-bottom: 1px solid {blend(fill, text, 0.70).name()};
            padding: 6px 8px 7px 8px;
            selection-background-color: palette(highlight);
            selection-color: palette(highlighted-text);
        }}
        QFrame[formInset="true"] QLineEdit:enabled:hover {{
            background-color: {blend(fill, text, 0.08).name()};
            border-bottom-color: {text.name()};
        }}
        QFrame[formInset="true"] QLineEdit:enabled:focus {{
            border-bottom: 2px solid palette(highlight);
            padding-bottom: 6px;
        }}
        QFrame[formInset="true"] QLineEdit:disabled {{
            background-color: {disabled_fill.name()};
            color: {disabled_text.name()};
            border-bottom-color: {disabled_text.name()};
        }}
        QPlainTextEdit[textPreview="true"]:enabled {{
            background-color: {surface.name()};
            border: 1px solid {border.name()};
        }}
        QPlainTextEdit[textPreview="true"]:enabled:hover {{
            border-color: palette(highlight);
        }}
    """


def apply_theme(app: QApplication) -> None:
    # Scheme selection belongs to ThemeController; styling must not turn a
    # system-following preference into an explicit light-mode override.
    is_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark

    # Qt's Windows dark checkbox palette uses the accent color for Base. Fusion
    # fills even unchecked indicators with Base; override only that state's fill.
    stylesheet = _BASE_STYLESHEET + (_DARK_STYLESHEET if is_dark else _LIGHT_STYLESHEET) + _form_stylesheet(app.palette())
    if app.styleSheet() != stylesheet:
        app.setStyleSheet(stylesheet)


class ThemeController(QObject):
    preference_changed = Signal(str)

    SCHEMES = {
        "dark": Qt.ColorScheme.Dark,
        "light": Qt.ColorScheme.Light,
        "system": Qt.ColorScheme.Unknown,
    }

    def __init__(self, app: QApplication, settings: QSettings):
        super().__init__(app)
        self.app = app
        self.settings = settings
        preference = settings.value("appearance/theme", "system")
        self.preference = preference if isinstance(preference, str) and preference in self.SCHEMES else "system"
        # Qt emits colorSchemeChanged before updating its palette. Refresh our
        # existing styling on the next event-loop turn, without changing the preference.
        app.styleHints().colorSchemeChanged.connect(self._refresh_theme, Qt.ConnectionType.QueuedConnection)
        app.paletteChanged.connect(self._refresh_theme, Qt.ConnectionType.QueuedConnection)
        self._apply_preference()

    def _apply_preference(self):
        scheme = Qt.ColorScheme.Dark if OT_FORCE_DARK else self.SCHEMES[self.preference]
        self.app.styleHints().setColorScheme(scheme)
        self._refresh_theme()

    @Slot()
    def _refresh_theme(self):
        apply_theme(self.app)

    @Slot(str)
    def set_preference(self, preference: str):
        if preference not in self.SCHEMES or preference == self.preference:
            return
        self.preference = preference
        self.settings.setValue("appearance/theme", preference)
        self.settings.sync()
        self._apply_preference()
        self.preference_changed.emit(preference)

    @Slot()
    def toggle(self):
        is_dark = self.app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        self.set_preference("light" if is_dark else "dark")
