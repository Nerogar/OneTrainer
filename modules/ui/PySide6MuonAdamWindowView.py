from modules.ui.BaseMuonAdamWindowView import BaseMuonAdamWindowView
from modules.ui.MuonAdamWindowController import MuonAdamWindowController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton


class PySide6MuonAdamWindowView(BaseMuonAdamWindowView, QDialog):
    def __init__(self, parent, controller: MuonAdamWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseMuonAdamWindowView.__init__(self, pyside6_components)

        self.setWindowTitle(controller.get_title())
        self.resize(800, 500)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll, frame = pyside6_components.scrollable_frame(self)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnMinimumWidth(2, 50)
        lo.setColumnStretch(4, 1)
        self.build_content(frame, controller, ui_state)
        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
