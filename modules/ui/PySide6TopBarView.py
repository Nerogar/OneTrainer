from collections.abc import Callable

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.i18n import LANGUAGES, get_language, set_language, t
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


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

        self._build_language_selector()

    def _build_language_selector(self):
        combo = pyside6_components.NoScrollComboBox(self.frame)
        codes = list(LANGUAGES.keys())
        combo.addItems([LANGUAGES[c] for c in codes])
        combo.setCurrentIndex(codes.index(get_language()))
        combo.setToolTip("Language / 語言")

        def on_change(index: int):
            set_language(codes[index])
            QMessageBox.information(
                self, "Language / 語言",
                "The language will change after restarting OneTrainer.\n重新啟動 OneTrainer 後套用新語言。",
            )

        combo.currentIndexChanged.connect(on_change)
        pyside6_components._add(pyside6_components._layout(self.frame), combo, 0, 8, sticky="vew")

    def _setup_frame_column_weight(self):
        pyside6_components._layout(self.frame).setColumnStretch(5, 1)

    def _forget_dropdown(self, widget):
        lo = pyside6_components._layout(self.frame)
        lo.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def _show_save_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getSaveFileName(self, t("Save config"), initial_dir, "JSON (*.json)")
        if path:
            # the native dialog doesn't reliably append the filter's extension on every platform
            if not path.endswith(".json"):
                path += ".json"
            callback(path)

    def _show_open_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getOpenFileName(self, t("Load config"), initial_dir, "JSON (*.json)")
        if path:
            callback(path)
