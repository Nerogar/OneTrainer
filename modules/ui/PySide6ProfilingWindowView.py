import contextlib

from modules.ui.BaseProfilingWindowView import BaseProfilingWindowView
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.util.ui import pyside6_components

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QWidget


class PySide6ProfilingWindowView(BaseProfilingWindowView, QWidget):
    def __init__(self, parent, controller: ProfilingWindowController):
        QWidget.__init__(self, parent, Qt.WindowType.Window)
        BaseProfilingWindowView.__init__(self, pyside6_components)

        self._controller = controller

        self.setWindowTitle("Profiling")
        self.resize(512, 512)

        outer = QGridLayout(self)
        outer.setRowStretch(2, 1)

        self._bottom_bar = QWidget(self)
        QGridLayout(self._bottom_bar)
        outer.addWidget(self._bottom_bar, 3, 0)

        self.build_content(self, self._bottom_bar, controller)

    def set_message(self, text: str):
        self._message_label.setText(text)

    def set_profiling_active(self, active: bool):
        if active:
            self._message_label.setText("Profiling active...")
            self._profile_button.setText("End Profiling")
            with contextlib.suppress(RuntimeError):
                self._profile_button.clicked.disconnect()
            self._profile_button.clicked.connect(self._controller.end_profiler)
        else:
            self._message_label.setText("Inactive")
            self._profile_button.setText("Start Profiling")
            with contextlib.suppress(RuntimeError):
                self._profile_button.clicked.disconnect()
            self._profile_button.clicked.connect(self._controller.start_profiler)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
