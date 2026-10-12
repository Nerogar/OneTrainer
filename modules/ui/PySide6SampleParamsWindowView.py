from modules.ui.BaseSampleParamsWindowView import BaseSampleParamsWindowView
from modules.ui.PySide6SampleFrameView import PySide6SampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.ui.SampleParamsWindowController import SampleParamsWindowController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QWidget


class PySide6SampleParamsWindowView(BaseSampleParamsWindowView, QDialog):
    def __init__(self, parent, controller: SampleParamsWindowController, ui_state):
        QDialog.__init__(self, parent if isinstance(parent, QWidget) else None)
        BaseSampleParamsWindowView.__init__(self, pyside6_components)

        self.setWindowTitle("Sample")
        self.resize(800, 500)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)
        outer.setColumnStretch(0, 1)

        frame = PySide6SampleFrameView(self, SampleFrameController(controller.sample, controller.model_type), ui_state)
        outer.addWidget(frame, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
