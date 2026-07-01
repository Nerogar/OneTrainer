from modules.ui.BaseCloudTabView import BaseCloudTabView
from modules.ui.CloudTabController import CloudTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QWidget


class PySide6CloudTabView(BaseCloudTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: CloudTabController, ui_state):
        QWidget.__init__(self, master)
        BaseCloudTabView.__init__(self, pyside6_components, controller)

        self.ui_state = ui_state

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        lo.setColumnStretch(5, 1)
        self.frame = frame

        self.build_content(frame, controller, ui_state)

    def _on_set_gpu_types(self):
        self.gpu_types_menu.clear()
        self.gpu_types_menu.addItems(self.controller.get_gpu_types())

    def _make_reattach_frame(self, frame):
        reattach_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(reattach_frame, 9, 3)
        pyside6_components._layout(reattach_frame).setColumnStretch(0, 1)
        return reattach_frame

    def _make_create_frame(self, frame):
        create_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(create_frame, 1, 5)
        pyside6_components._layout(create_frame).setColumnStretch(1, 1)
        return create_frame
