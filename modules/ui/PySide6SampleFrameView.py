from modules.ui.BaseSampleFrameView import BaseSampleFrameView
from modules.ui.SampleFrameController import SampleFrameController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QWidget


class PySide6SampleFrameView(BaseSampleFrameView, QWidget):
    def __init__(
            self,
            parent: QWidget,
            controller: SampleFrameController,
            ui_state,
            include_prompt: bool = True,
            include_settings: bool = True,
    ):
        QWidget.__init__(self, parent)
        BaseSampleFrameView.__init__(self, pyside6_components)

        outer = pyside6_components._layout(self)
        outer.setColumnStretch(0, 1)
        if include_prompt and include_settings:
            outer.setRowStretch(1, 1)

        top_frame = None
        if include_prompt:
            top_frame = QWidget(self)
            top_lo = pyside6_components._layout(top_frame)
            top_lo.setColumnStretch(1, 1)
            outer.addWidget(top_frame, 0, 0)

        bottom_frame = None
        if include_settings:
            bottom_frame = QWidget(self)
            bot_lo = pyside6_components._layout(bottom_frame)
            bot_lo.setColumnStretch(1, 1)
            bot_lo.setColumnStretch(3, 1)
            row = 1 if include_prompt else 0
            outer.addWidget(bottom_frame, row, 0)

        self.build_content(top_frame, bottom_frame, ui_state, controller, include_prompt, include_settings)

        if top_frame is not None:
            pyside6_components._pack_form(top_frame)
        if bottom_frame is not None:
            pyside6_components._pack_form(bottom_frame)
