from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget


class MainWindowSettings:
    """Persist the training window independently of training presets."""

    def __init__(
            self,
            window: QMainWindow,
            tabs: QTabWidget,
            tab_widgets: dict[str, QWidget],
            settings: QSettings,
    ):
        self.window = window
        self.tabs = tabs
        self.tab_widgets = tab_widgets
        self.settings = settings

    def restore(self):
        geometry = self.settings.value("mainWindow/geometry")
        if isinstance(geometry, QByteArray):
            self.window.restoreGeometry(geometry)

        key = self.settings.value("mainWindow/lastTab", "general")
        page = self.tab_widgets.get(key) if isinstance(key, str) else None
        index = self.tabs.indexOf(page) if page is not None else -1
        if index < 0 or not self.tabs.isTabVisible(index) or not self.tabs.isTabEnabled(index):
            page = self.tab_widgets["general"]
        self.tabs.setCurrentWidget(page)
        # Startup tab construction/restoration must not overwrite the saved tab.
        self.tabs.currentChanged.connect(self.save_last_tab)

    def save_last_tab(self, _index: int = -1):
        index = self.tabs.currentIndex()
        if index < 0 or not self.tabs.isTabVisible(index):
            return
        current = self.tabs.currentWidget()
        for key, page in self.tab_widgets.items():
            if page is current:
                self.settings.setValue("mainWindow/lastTab", key)
                break

    def save(self):
        self.settings.setValue("mainWindow/geometry", self.window.saveGeometry())
        self.save_last_tab()
        self.settings.sync()
