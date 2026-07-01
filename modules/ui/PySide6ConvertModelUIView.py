from modules.ui.BaseConvertModelUIView import BaseConvertModelUIView
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QDialog, QGridLayout, QWidget


class PySide6ConvertModelUIView(BaseConvertModelUIView, QDialog):
    def __init__(self, parent, controller: ConvertModelUIController):
        QDialog.__init__(self, parent)
        BaseConvertModelUIView.__init__(self, pyside6_components)

        ui_state = PySide6UIState(controller.convert_model_args)

        self.setWindowTitle("Convert models")
        self.resize(600, 380)

        _pad = pyside6_components.PAD
        outer = QGridLayout(self)
        outer.setContentsMargins(_pad, _pad, _pad, _pad)

        frame = QWidget(self)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        outer.addWidget(frame, 0, 0)

        self.build_content(frame, controller, ui_state)
        lo.setRowStretch(lo.rowCount(), 1)

    def set_converting(self, active):
        self.button.setEnabled(not active)
