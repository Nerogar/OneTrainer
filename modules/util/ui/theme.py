import os

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Lets reviewers whose OS never reports dark still see the dark theme, by pretending it did.
OT_FORCE_DARK = os.environ.get("OT_FORCE_DARK") == "1"

_DARK_CHECKBOX_CSS = """
    QCheckBox::indicator:unchecked {
        width: 14px; height: 14px; background-color: palette(window);
        border: 1px solid palette(midlight); border-radius: 2px;
    }
    QCheckBox[largeIndicator="true"]::indicator:unchecked { width: 18px; height: 18px; }
    QCheckBox::indicator:unchecked:enabled:hover, QCheckBox::indicator:unchecked:enabled:focus { border-color: palette(text); }
    QCheckBox::indicator:unchecked:enabled:pressed { background-color: palette(mid); }
    QCheckBox::indicator:unchecked:disabled { border-color: palette(mid); }
"""

_LIGHT_TOKENS = {
    "input_bg": "white", "input_border": "#c8c8c8", "input_disabled_bg": "#e0e0e0", "input_disabled_fg": "#787878",
    "text": "#000000", "field": "#1d1d1d", "surface": "#fafafa", "border": "#d3d3d3",
    "inset": "#f4f4f4", "inset_border": "#c3c3c3", "fill": "#e5e5e5", "fill_border": "#454545",
    "hover_fill": "#d3d3d3", "disabled_fill": "#eaeaea", "disabled_text": "#919191",
    "extra": "QProgressBar { background-color: #c8c8c8; }",
}

_DARK_TOKENS = {
    "input_bg": "#404040", "input_border": "#242424", "input_disabled_bg": "#282828", "input_disabled_fg": "#808080",
    "text": "#ffffff", "field": "#e4e4e4", "surface": "#2d2d2d", "border": "#3b3b3b",
    "inset": "#282828", "inset_border": "#535353", "fill": "#424242", "fill_border": "#c6c6c6",
    "hover_fill": "#515151", "disabled_fill": "#313131", "disabled_text": "#7f7f7f",
    "extra": _DARK_CHECKBOX_CSS,
}

_STYLESHEET = """
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        padding: 2px 2px; border-radius: 2px;
        background-color: {input_bg}; color: palette(text); border: 1px solid {input_border};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
        background-color: {input_disabled_bg}; color: {input_disabled_fg};
    }}
    QPushButton[class~="destructive-action"], QPushButton[class~="confirm-action"] {{ border: 1px solid transparent; border-radius: 2px; }}
    QPushButton[class~="destructive-action"] {{ background-color: #FFECEC; color: #5A0D0D; border-color: #620303; }}
    QPushButton[class~="confirm-action"] {{ background-color: #D7F5EA; color: #0A4933; border-color: #045F3F; }}
    QPushButton[class~="destructive-action"]:enabled:hover {{ background-color: #FFD6D6; border-color: #520202; }}
    QPushButton[class~="destructive-action"]:enabled:pressed {{ background-color: #FFB8B8; color: #470808; border-color: #3D0101; }}
    QPushButton[class~="confirm-action"]:enabled:hover {{ background-color: #BBEBD9; border-color: #034D33; }}
    QPushButton[class~="confirm-action"]:enabled:pressed {{ background-color: #9CDFC6; color: #063826; border-color: #023B27; }}
    QPushButton[class~="destructive-action"]:focus, QPushButton[class~="confirm-action"]:focus {{ border-style: dotted; }}
    QPushButton[compactButton="true"] {{ padding: 3px 8px; }}
    QPushButton#trainingButton, QPushButton#tensorboardButton {{ font-size: 10pt; }}
    QPushButton#trainingButton {{ padding: 3px 10px; font-weight: 550; }}
    QPushButton[listAction="remove"], QPushButton[listAction="clone"] {{ padding: 0; font-size: 18px; }}
    QPushButton[listAction="clone"] {{ font-weight: bold; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; }}
    QCheckBox[largeIndicator="true"]::indicator {{ width: 20px; height: 20px; }}
    QToolButton {{ padding-top: 0px; padding-bottom: 0px; padding-right: 40px; }}
    QToolButton::menu-indicator {{ subcontrol-origin: padding; subcontrol-position: right center; width: 12px; height: 12px; right: 10px; }}
    QLabel[formLabel="heading"]:enabled {{ color: {text}; }}
    QLabel[formLabel="field"]:enabled {{ color: {field}; }}
    QFrame[formPanel="true"] {{ background-color: {surface}; border: 1px solid {border}; border-radius: 6px; }}
    QFrame[formInset="true"] {{ background-color: {inset}; border: 1px solid {inset_border}; border-radius: 6px; }}
    QFrame[formInset="true"] QLineEdit {{
        background-color: {fill}; color: {text}; border: none; border-bottom: 1px solid {fill_border};
        padding: 6px 8px 7px 8px; selection-background-color: palette(highlight); selection-color: palette(highlighted-text);
    }}
    QFrame[formInset="true"] QLineEdit:enabled:hover {{ background-color: {hover_fill}; border-bottom-color: {text}; }}
    QFrame[formInset="true"] QLineEdit:enabled:focus {{ border-bottom: 2px solid palette(highlight); padding-bottom: 6px; }}
    QFrame[formInset="true"] QLineEdit:disabled {{ background-color: {disabled_fill}; color: {disabled_text}; border-bottom-color: {disabled_text}; }}
    QPlainTextEdit[textPreview="true"]:enabled {{ background-color: {surface}; border: 1px solid {border}; }}
    QPlainTextEdit[textPreview="true"]:enabled:hover {{ border-color: palette(highlight); }}
    {extra}
"""


def _dark_palettes() -> tuple[QPalette, QPalette]:
    """Windows Dark palette forced on all platforms for consistent contrast."""
    r, disabled = QPalette.ColorRole, QPalette.ColorGroup.Disabled
    p = QPalette(QColor("#3c3c3c"), QColor("#1e1e1e"))
    for role, color in (
        (r.Light, "#787878"), (r.Midlight, "#5a5a5a"), (r.BrightText, "#99ebff"),
        (r.Base, "#2d2d2d"), (r.AlternateBase, "#262626"), (r.Highlight, "#0078d4"),
        (r.Link, "#99ebff"), (r.LinkVisited, "#4cc2ff"),
        (r.ToolTipBase, "#3c3c3c"), (r.ToolTipText, "#d4d4d4"), (r.Accent, "#4cc2ff"),
    ):
        p.setColor(role, QColor(color))
    for role in (r.WindowText, r.Text, r.ButtonText, r.Accent):
        p.setColor(disabled, role, QColor("#9d9d9d"))
    p.setColor(disabled, r.Base, QColor("#1e1e1e"))
    p.setColor(r.PlaceholderText, QColor(255, 255, 255, 128))
    # Checked indicators use QCheckBox's Base color in Fusion; unchecked is handled in QSS.
    cb = QPalette(p)
    cb.setColor(r.Base, QColor("#0078d4"))
    cb.setColor(disabled, r.Base, QColor("#1e1e1e"))
    return p, cb


def apply_theme(app: QApplication) -> None:
    # Scheme selection belongs to ThemeController; styling must not turn a
    # system-following preference into an explicit light-mode override.
    is_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    if is_dark:
        palette, checkbox_palette = _dark_palettes()
        app.setPalette(palette)
        app.setPalette(checkbox_palette, "QCheckBox")
        app.setProperty("_ot_dark_palette", True)
    elif app.property("_ot_dark_palette"):
        app.setProperty("_ot_dark_palette", False)
        app.setPalette(QPalette())

    stylesheet = _STYLESHEET.format(**(_DARK_TOKENS if is_dark else _LIGHT_TOKENS))
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
