from modules.ui.BaseLoraTabView import BaseLoraTabView
from modules.ui.LoraTabController import LoraTabController
from modules.util.enum.ModelType import PeftType
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QWidget


class PySide6LoraTabView(BaseLoraTabView, QWidget):

    def __init__(self, master, controller: LoraTabController, ui_state):
        QWidget.__init__(self, master)
        BaseLoraTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self.options_frame = None
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        self.scroll_frame = QWidget(self)
        pyside6_components._layout(self).addWidget(self.scroll_frame, 0, 0)
        lo = pyside6_components._layout(self.scroll_frame)
        lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(2, 2)
        self.build(self.scroll_frame, self.controller, self.ui_state, self.setup_lora)
        pyside6_components._pack_form(self.scroll_frame)

    def setup_lora(self, peft_type: PeftType):
        if self.options_frame is not None:
            self.options_frame.hide()
            self.options_frame.deleteLater()

        self.options_frame = QWidget(self.scroll_frame)
        pyside6_components._layout(self.scroll_frame).addWidget(self.options_frame, 1, 0, 1, 3)
        lo = pyside6_components._layout(self.options_frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(4, 1)
        self.build_lora_options(self.options_frame, self.controller, self.ui_state, peft_type)
        pyside6_components._pack_form(self.options_frame)
