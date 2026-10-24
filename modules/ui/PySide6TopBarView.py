from collections.abc import Callable

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QFileDialog, QWidget


class PySide6TopBarView(BaseTopBarView, QWidget):

    def __init__(
            self,
            master,
            controller: TopBarController,
            ui_state,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        QWidget.__init__(self, master)
        BaseTopBarView.__init__(self, pyside6_components)

        self.frame = QWidget(self)
        pyside6_components._layout(self).addWidget(self.frame, 0, 0)
        pyside6_components._layout(self.frame).setContentsMargins(
            pyside6_components.PAD, pyside6_components.PAD,
            pyside6_components.PAD, pyside6_components.PAD,
        )

        self.build(self.frame, master, controller, ui_state,
                   change_model_type_callback, change_training_method_callback, load_preset_callback)

    def _setup_frame_column_weight(self):
        pyside6_components._layout(self.frame).setColumnStretch(5, 1)

    def _forget_dropdown(self, widget):
        lo = pyside6_components._layout(self.frame)
        lo.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def _show_save_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", initial_dir, "JSON (*.json)")
        if path:
            # the native dialog doesn't reliably append the filter's extension on every platform
            if not path.endswith(".json"):
                path += ".json"
            callback(path)

    def _show_open_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", initial_dir, "JSON (*.json)")
        if path:
            callback(path)
