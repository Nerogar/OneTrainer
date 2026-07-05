import os
from abc import abstractmethod
from collections.abc import Callable

from modules.util import path_util
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer


def preset_name_from_filename(filename: str) -> str | None:
    # The name we show in the top bar. A preset or config is shown by its file
    # name without the ".json" ending. "#.json" is the auto-saved "last session"
    # file, not a real preset, so we show nothing for it.
    basename = os.path.basename(filename)
    if basename == "#.json":
        return None
    return os.path.splitext(basename)[0]


class BaseTopBarView:
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def _setup_frame_column_weight(self):
        pass

    @abstractmethod
    def _forget_dropdown(self, widget):
        pass

    @abstractmethod
    def _show_save_dialog(self, initial_dir: str, callback):
        pass

    @abstractmethod
    def _show_open_dialog(self, initial_dir: str, callback):
        pass

    # --- preset-name display hooks -----------------------------------------
    # No-ops by default. The PySide6 top bar overrides them to show the loaded
    # preset/config name on the "Load Preset" button; the Ctk top bar leaves
    # the button showing its static text.

    def _init_preset_display(self):
        pass

    def _set_preset_text(self, text: str):
        pass

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
        loaded_config = self.controller.load_config_from_file(filename)
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
