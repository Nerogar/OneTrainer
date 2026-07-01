import json
import os
import traceback
from abc import abstractmethod
from collections.abc import Callable
from contextlib import suppress

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer


class BaseTopBarView:
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def _make_config_ui_state(self, master, data):
        pass

    @abstractmethod
    def _get_dropdown_text(self, widget) -> str:
        pass

    @abstractmethod
    def _setup_frame_column_weight(self):
        pass

    @abstractmethod
    def _forget_dropdown(self, widget):
        pass

    @abstractmethod
    def _show_save_dialog(self, default_value: str, callback):
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

        self.config_ui_data = {
            "config_name": path_util.canonical_join(self.dir, "#.json")
        }
        self.config_ui_state = self._make_config_ui_state(master, self.config_ui_data)

        self.configs = controller.load_available_config_names(self.dir)

        self.current_config = []

        self.training_method = None

        # title
        self.components.app_title(self.frame, 0, 0)

        # dropdown
        self.configs_dropdown = None
        self.__create_configs_dropdown()

        # remove button
        # TODO
        # self.components.icon_button(self.frame, 0, 2, "-", self.__remove_config)

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

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self._forget_dropdown(self.configs_dropdown)

        self.configs_dropdown = self.components.options_kv(
            self.frame, 0, 1, self.configs, self.config_ui_state, "config_name", self.__load_current_config,
            sticky="vew",
        )

    def __save_config(self):
        default_value = self._get_dropdown_text(self.configs_dropdown)
        while default_value.startswith('#'):
            default_value = default_value[1:]

        self._show_save_dialog(default_value, self.__save_new_config)

    def __save_new_config(self, name):
        path = self.controller.save_to_file(name)

        is_new_config = name not in [x[0] for x in self.configs]

        if is_new_config:
            self.configs.append((name, path))
            self.configs.sort()

        if self.config_ui_data["config_name"] != path_util.canonical_join(self.dir, f"{name}.json"):
            self.config_ui_state.get_var("config_name").set(path_util.canonical_join(self.dir, f"{name}.json"))

        if is_new_config:
            self.__create_configs_dropdown()

    def __load_current_config(self, filename):
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                # built-in configs are always saved in the most recent version, so migration can be skipped
                loaded_config = default_config.from_dict(loaded_dict, migrate=not is_built_in_preset).to_unpacked_config()

            with suppress(FileNotFoundError), open("secrets.json", "r") as f:
                secrets_dict=json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.controller.train_config.from_dict(loaded_config.to_dict())
            self.ui_state.update(loaded_config)

            optimizer_config = change_optimizer(self.controller.train_config)
            self.ui_state.get_var("optimizer").update(optimizer_config)

            self.load_preset_callback()
        except FileNotFoundError:
            pass
        except Exception:
            print(traceback.format_exc())

    def __remove_config(self):
        # TODO
        pass

    def open_wiki(self):
        self.controller.open_wiki()

    def save_default(self):
        self.controller.save_default()
