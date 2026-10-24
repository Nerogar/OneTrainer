from modules.ui.BaseModelTabView import BaseModelTabView
from modules.ui.ModelTabController import ModelTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QWidget


class PySide6ModelTabView(BaseModelTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: ModelTabController, ui_state):
        QWidget.__init__(self, master)
        BaseModelTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self.refresh_ui()

    def _make_svd_frames(self, parent, row: int):
        svd_label_frame = QWidget(parent)
        pyside6_components._layout(parent).addWidget(svd_label_frame, row, 3)
        svd_entry_frame = QWidget(parent)
        pyside6_components._layout(parent).addWidget(svd_entry_frame, row, 4)
        return svd_label_frame, svd_entry_frame

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        self.scroll_frame = scroll

        frame_lo = pyside6_components._layout(frame)
        frame_lo.setColumnStretch(1, 10)
        frame_lo.setColumnStretch(4, 1)

        self.build_content(frame, self.controller, self.ui_state)
