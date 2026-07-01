from modules.ui.BaseOptimizerParamsWindowView import BaseOptimizerParamsWindowView
from modules.ui.MuonAdamWindowController import MuonAdamWindowController
from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.ui.PySide6MuonAdamWindowView import PySide6MuonAdamWindowView
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QWidget


class PySide6OptimizerParamsWindowView(BaseOptimizerParamsWindowView, QDialog):
    def __init__(self, parent, controller: OptimizerParamsWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseOptimizerParamsWindowView.__init__(self, pyside6_components)

        self.controller = controller
        self.ui_state = ui_state
        self.optimizer_ui_state = ui_state.get_var("optimizer")
        self.muon_adam_button = None
        self._dynamic_frame = None

        self.setWindowTitle("Optimizer Settings")
        self.resize(800, 500)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll, self._frame = pyside6_components.scrollable_frame(self)
        lo = pyside6_components._layout(self._frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnMinimumWidth(2, 50)
        lo.setColumnStretch(4, 1)
        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self._on_close)
        outer.addWidget(ok, 1, 0)

        self.build_content(self._frame, controller, ui_state, self.optimizer_ui_state,
                           self.on_optimizer_change, self._load_defaults)
        self._rebuild_dynamic_ui()


    def _rebuild_dynamic_ui(self):
        if self._dynamic_frame is not None:
            self._dynamic_frame.hide()
            self._dynamic_frame.deleteLater()

        self._dynamic_frame = QWidget(self._frame)
        pyside6_components._layout(self._frame).addWidget(self._dynamic_frame, 1, 0, 1, 5)

        self.build_dynamic_content(self._dynamic_frame, self.controller, self.optimizer_ui_state,
                                   self.update_user_pref, self.open_muon_adam_window)
        self.toggle_muon_adam_button()

    def update_user_pref(self, *args):
        self.controller.on_close()
        self.toggle_muon_adam_button()

    def on_optimizer_change(self, *args):
        self.controller.restore_optimizer_config(self.ui_state)
        self._rebuild_dynamic_ui()

    def _load_defaults(self, *args):
        self.controller.load_defaults(self.ui_state)

    def _on_close(self):
        self.controller.on_close()
        self.accept()

    def toggle_muon_adam_button(self):
        if self.muon_adam_button is not None:
            muon_with_adam = self.optimizer_ui_state.get_var("MuonWithAuxAdam").get()
            self.muon_adam_button.setEnabled(bool(muon_with_adam))

    def open_muon_adam_window(self):
        adam_config, current_optimizer = self.controller.prepare_muon_adam_config()
        adam_ui_state = PySide6UIState(adam_config)
        PySide6MuonAdamWindowView(self, MuonAdamWindowController(self.controller.config, current_optimizer), adam_ui_state).exec()
        self.controller.save_muon_adam_config(adam_config)
