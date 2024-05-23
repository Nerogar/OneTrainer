import json
import os
import traceback
from typing import Callable

import customtkinter as ctk

from modules.util import path_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState


class TopBar:
    def __init__(
            self,
            master,
            train_config: TrainConfig,
            ui_state: UIState,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.change_model_type_callback = change_model_type_callback
        self.change_training_method_callback = change_training_method_callback
        self.load_preset_callback = load_preset_callback

        self.dir = "training_presets"

        self.config_ui_data = {
            "config_name": path_util.canonical_join(self.dir, "#.json")
        }
        self.config_ui_state = UIState(master, self.config_ui_data)

        self.configs = [("", path_util.canonical_join(self.dir, "#.json"))]
        self.__load_available_config_names()

        self.current_config = []

        self.frame = ctk.CTkFrame(master=master, corner_radius=0)
        self.frame.grid(row=0, column=0, sticky="nsew")

        self.training_method = None

        # title
        components.app_title(self.frame, 0, 0)

        # dropdown
        self.configs_dropdown = None
        self.__create_configs_dropdown()

        # remove button
        # TODO
        # components.icon_button(self.frame, 0, 2, "-", self.__remove_config)

        # save button
        components.button(self.frame, 0, 3, "save current config", self.__save_config,
                          tooltip="Save the current configuration in a custom preset")

        # padding
        self.frame.grid_columnconfigure(4, weight=1)

        # model type
        components.options_kv(
            master=self.frame,
            row=0,
            column=5,
            values=[
                ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
                ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
                ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
                ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
                ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
                ("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE),
                ("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
                ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
                ("Stable Cascade", ModelType.STABLE_CASCADE_1),
                ("PixArt Alpha", ModelType.PIXART_ALPHA),
                ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ],
            ui_state=self.ui_state,
            var_name="model_type",
            command=self.__change_model_type,
        )

    def __create_training_method(self):
        if self.training_method:
            self.training_method.destroy()

        values = []

        if self.train_config.model_type.is_stable_diffusion():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ]
        elif self.train_config.model_type.is_stable_diffusion_xl():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]
        elif self.train_config.model_type.is_wuerstchen():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]
        elif self.train_config.model_type.is_pixart():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]

        # training method
        self.training_method = components.options_kv(
            master=self.frame,
            row=0,
            column=6,
            values=values,
            ui_state=self.ui_state,
            var_name="training_method",
            command=self.change_training_method_callback,
        )

    def __change_model_type(self, model_type: ModelType):
        self.change_model_type_callback(model_type)
        self.__create_training_method()

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.grid_forget()

        self.configs_dropdown = components.options_kv(
            self.frame, 0, 1, self.configs, self.config_ui_state, "config_name", self.__load_current_config
        )

    def __load_available_config_names(self):
        if os.path.isdir(self.dir):
            for path in os.listdir(self.dir):
                if path != "#.json":
                    path = path_util.canonical_join(self.dir, path)
                    if path.endswith(".json") and os.path.isfile(path):
                        name = os.path.basename(path)
                        name = os.path.splitext(name)[0]
                        self.configs.append((name, path))

    def __save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        with open(path, "w") as f:
            json.dump(self.train_config.to_dict(), f, indent=4)

        return path

    def __save_new_config(self, name):
        path = self.__save_to_file(name)

        is_new_config = name not in [x[0] for x in self.configs]

        if is_new_config:
            self.configs.append((name, path))

        if self.config_ui_data["config_name"] != path_util.canonical_join(self.dir, f"{name}.json"):
            self.config_ui_state.get_var("config_name").set(path_util.canonical_join(self.dir, f"{name}.json"))

        if is_new_config:
            self.__create_configs_dropdown()

    def __save_config(self):
        default_value = self.configs_dropdown.get()
        while default_value.startswith('#'):
            default_value = default_value[1:]

        dialogs.StringInputDialog(
            parent=self.master,
            title="name",
            question="Config Name",
            callback=self.__save_new_config,
            default_value=default_value,
            validate_callback=lambda x: not x.startswith("#")
        )

    def __load_current_config(self, filename):
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                if is_built_in_preset:
                    # always assume built-in configs are saved in the most recent version
                    loaded_dict["__version"] = default_config.config_version
                loaded_config = default_config.from_dict(loaded_dict).to_unpacked_config()

            self.train_config.from_dict(loaded_config.to_dict())
            self.ui_state.update(loaded_config)

            optimizer_config = change_optimizer(self.train_config)
            self.ui_state.get_var("optimizer").update(optimizer_config)

            self.load_preset_callback()
        except Exception:
            print(traceback.format_exc())

    def __remove_config(self):
        # TODO
        pass

    def save_default(self):
        self.__save_to_file("#")
