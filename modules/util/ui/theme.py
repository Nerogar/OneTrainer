import os

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
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
    # Scheme selection belongs to ThemeController; styling must not turn a
    # system-following preference into an explicit light-mode override.
    is_dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark

    palette = app.palette()
    if not is_dark:
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e0e0e0"))
    app.setPalette(palette)
    app.setStyleSheet(_BASE_STYLESHEET if is_dark else _BASE_STYLESHEET + _LIGHT_STYLESHEET)


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
