from modules.ui.BaseTrainingTabView import BaseTrainingTabView
from modules.ui.OffloadingWindowController import OffloadingWindowController
from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.ui.PySide6OffloadingWindowView import PySide6OffloadingWindowView
from modules.ui.PySide6OptimizerParamsWindowView import PySide6OptimizerParamsWindowView
from modules.ui.PySide6SchedulerParamsWindowView import PySide6SchedulerParamsWindowView
from modules.ui.PySide6TimestepDistributionWindowView import PySide6TimestepDistributionWindowView
from modules.ui.SchedulerParamsWindowController import SchedulerParamsWindowController
from modules.ui.TimestepDistributionWindowController import TimestepDistributionWindowController
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget


class PySide6TrainingTabView(BaseTrainingTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: TrainingTabController, ui_state):
        QWidget.__init__(self, master)
        BaseTrainingTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None
        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.hide()
            self.scroll_frame.deleteLater()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        self.scroll_frame = scroll

        frame = QWidget()
        scroll.setWidget(frame)

        lo = pyside6_components._layout(frame)
        lo.setContentsMargins(pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD, pyside6_components.PAD)
        lo.setColumnStretch(0, 1)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(2, 1)

        column_0 = QWidget(frame)
        column_0.setMinimumWidth(0)
        column_0.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(frame).addWidget(column_0, 0, 0)
        pyside6_components._layout(column_0).setColumnStretch(0, 1)

        column_1 = QWidget(frame)
        column_1.setMinimumWidth(0)
        column_1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(frame).addWidget(column_1, 0, 1)
        pyside6_components._layout(column_1).setColumnStretch(0, 1)

        column_2 = QWidget(frame)
        column_2.setMinimumWidth(0)
        column_2.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pyside6_components._layout(frame).addWidget(column_2, 0, 2)
        pyside6_components._layout(column_2).setColumnStretch(0, 1)

        self.build(column_0, column_1, column_2, self.controller, self.ui_state)

        for col_widget in (column_0, column_1, column_2):
            lo = pyside6_components._layout(col_widget)
            lo.setRowStretch(lo.rowCount(), 1)

    def restore_optimizer_config(self, variable: str):
        self.controller.restore_optimizer_config(self.ui_state)

    def restore_scheduler(self, variable: str):
        if not hasattr(self, 'lr_scheduler_adv_comp'):
            return
        self.lr_scheduler_adv_comp.setEnabled(self.controller.is_custom_scheduler_value(variable))

    def open_optimizer_params(self):
        PySide6OptimizerParamsWindowView(self, OptimizerParamsWindowController(self.controller.config), self.ui_state).exec()

    def open_scheduler_params(self):
        PySide6SchedulerParamsWindowView(self, SchedulerParamsWindowController(self.controller.config), self.ui_state).exec()

    def open_offloading(self):
        PySide6OffloadingWindowView(self, OffloadingWindowController(self.controller.config), self.ui_state).exec()

    def open_timestep_distribution(self):
        PySide6TimestepDistributionWindowView(self, TimestepDistributionWindowController(self.controller.config), self.ui_state).exec()
