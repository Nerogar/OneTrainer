from collections.abc import Callable

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QInputDialog, QMessageBox, QWidget


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

    def _make_config_ui_state(self, master, data):
        from modules.util.ui.PySide6UIState import PySide6UIState
        return PySide6UIState(data)

    def _get_dropdown_text(self, widget) -> str:
        return widget.currentText()

    def _setup_frame_column_weight(self):
        pyside6_components._layout(self.frame).setColumnStretch(5, 1)

    def _forget_dropdown(self, widget):
        lo = pyside6_components._layout(self.frame)
        lo.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def _show_save_dialog(self, default_value: str, callback):
        text, ok = QInputDialog.getText(self, "name", "Config Name", text=default_value)
        if not ok:
            return
        if text.startswith("#"):
            # names starting with '#' are reserved for built-in presets; warn
            # instead of silently discarding the save, so the user can retry
            QMessageBox.warning(self, "Invalid name", "Config names starting with '#' are reserved for built-in presets.")
            return
        callback(text)
