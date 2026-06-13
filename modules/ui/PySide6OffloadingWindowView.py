from modules.ui.BaseOffloadingWindowView import BaseOffloadingWindowView
from modules.ui.OffloadingWindowController import OffloadingWindowController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton


class PySide6OffloadingWindowView(BaseOffloadingWindowView, QDialog):
    def __init__(self, parent, controller: OffloadingWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseOffloadingWindowView.__init__(self, pyside6_components)

        self.setWindowTitle("Offloading")
        self.resize(800, 400)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(frame).setColumnStretch(0, 1)
        pyside6_components._layout(frame).setColumnStretch(1, 1)
        self.build_content(frame, controller, ui_state)
        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
