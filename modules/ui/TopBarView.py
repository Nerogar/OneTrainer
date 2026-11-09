import os
from collections.abc import Callable

from modules.ui.TopBarController import TopBarController
from modules.util import path_util
from modules.util.config.TrainConfig import RemovedOptimizerError
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import get_theme_controller

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QWidget


def preset_name_from_filename(filename: str) -> str | None:
    # Show the filename without its extension, dont for last-session "#.json".
    basename = os.path.basename(filename)
    if basename == "#.json":
        return None
    return os.path.splitext(basename)[0]


class TopBarView(QWidget):

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
        self.components = pyside6_components

        self.frame = QWidget(self)
        pyside6_components._layout(self).addWidget(self.frame, 0, 0)
        pyside6_components._layout(self.frame).setContentsMargins(
            pyside6_components.PAD, pyside6_components.PAD,
            pyside6_components.PAD, pyside6_components.PAD,
        )

        self.build(self.frame, master, controller, ui_state,
                   change_model_type_callback, change_training_method_callback, load_preset_callback)

        # Theme toggle; the glyph follows the palette's ButtonText.
        self.theme_button = pyside6_components.button(
            self.frame, 0, 8, "◐", get_theme_controller().toggle,
            tooltip="Switch between light and dark mode", sticky="v",
        )
        # Use an empty button's height for a fixed square size, independent of glyph metrics.
        side = QPushButton().sizeHint().height()
        self.theme_button.setFixedSize(side, side)

    def _setup_frame_column_weight(self):
        pyside6_components._layout(self.frame).setColumnStretch(5, 1)

    def _show_config_error(self, message: str):
        QMessageBox.critical(self, "Cannot Load Config", message)

    def _init_preset_display(self):
        # the "Load Preset" button doubles as the display of current preset
        self.preset_button.setToolTip("Load a preset")

    def _set_preset_text(self, text: str):
        # empty means nothing is loaded
        label = text or "Load Preset"
        # limit width
        label = self.preset_button.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, 240)
        self.preset_button.setText(label.replace("&", "&&"))
        self.preset_button.setToolTip(f"Load a preset\n{text}" if text else "Load a preset")

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

    def build(
            self,
            frame,
            master,
            controller,
            ui_state,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        self.controller = controller
        self.frame = frame
        self.master = master
        self.ui_state = ui_state
        self.change_model_type_callback = change_model_type_callback
        self.change_training_method_callback = change_training_method_callback
        self.load_preset_callback = load_preset_callback

        self.dir = "training_presets"

        self.preset_tree = controller.load_preset_tree(self.dir)

        self.current_config = []

        self.training_method = None

        # name of the preset/config currently loaded (None = nothing to show)
        self.current_preset_name = None

        # title
        self.components.app_title(self.frame, 0, 0)

        # preset picker: the button shows the loaded preset/config name and
        # opens the "model type -> presets" menu when clicked
        self.preset_button = self.components.preset_menu_button(
            self.frame, 0, 1, "Load Preset", self.preset_tree, self.__load_current_config, sticky="vew",
        )
        self._init_preset_display()

        # load config button
        self.components.button(self.frame, 0, 2, "Load config", self.__load_config,
                               tooltip="Load one of your own saved configs", width=90, sticky="vew")

        # Wiki button
        self.components.button(self.frame, 0, 4, "Wiki", self.open_wiki, width=50, sticky="vew")

        # save button
        self.components.button(self.frame, 0, 3, "Save config", self.__save_config,
                               tooltip="Save the current configuration in a custom preset", width=90, sticky="vew")

        # padding
        self._setup_frame_column_weight()

        # model type
        self.components.options_kv(
            master=self.frame,
            row=0,
            column=6,
            values=controller.get_model_types(),
            ui_state=ui_state,
            var_name="model_type",
            command=self.__change_model_type,
            sticky="vew",
        )

        # restore the config from the previous session
        self.__load_current_config(path_util.canonical_join(self.dir, "#.json"))

    def __create_training_method(self):
        if self.training_method:
            self._forget_dropdown(self.training_method)

        values = self.controller.get_training_methods(self.controller.train_config.model_type)

        self.training_method = self.components.options_kv(
            master=self.frame,
            row=0,
            column=7,
            values=values,
            ui_state=self.ui_state,
            var_name="training_method",
            command=self.change_training_method_callback,
            sticky="vew",
        )

    def __change_model_type(self, model_type: ModelType):
        self.change_model_type_callback(model_type)
        self.__create_training_method()

    def __load_config(self):
        self._show_open_dialog("training_configs", self.__load_current_config)

    def __save_config(self):
        # after the file is written, remember it as the current preset so the
        # name shows in the top bar and the "changed" marker clears
        def save_and_remember(path):
            self.controller.save_config_to_path(path)
            self.__set_current_preset(path)

        self._show_save_dialog("training_configs", save_and_remember)

    def __load_current_config(self, filename):
        try:
            loaded_config = self.controller.load_config_from_file(filename)
        except RemovedOptimizerError as error:
            self._show_config_error(str(error))
            return
        if loaded_config is None:
            return

        self.ui_state.update(loaded_config)

        optimizer_config = change_optimizer(self.controller.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

        self.__set_current_preset(filename)

        self.load_preset_callback()

    def __set_current_preset(self, filename):
        # record which preset/config is now loaded, then update the label
        self.current_preset_name = preset_name_from_filename(filename)
        self._set_preset_text(self.current_preset_name or "")

    def __remove_config(self):
        # TODO
        pass

    def open_wiki(self):
        self.controller.open_wiki()

    def save_default(self):
        self.controller.save_default()
