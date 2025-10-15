import json
import os
import traceback
import webbrowser
from collections.abc import Callable
from contextlib import suppress

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.path_util import write_json_atomic
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState

import customtkinter as ctk


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
        self._topbar_wrapped = False
        self._resize_pending = False

        # title stays alone on the left
        components.app_title(self.frame, 0, 0)

        self.toolbar = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.toolbar.grid(row=0, column=1, sticky="nsew")

        self.frame.grid_columnconfigure(1, weight=1)

        # left side of toolbar
        self.configs_dropdown = None
        self.__create_configs_dropdown()

        self.save_btn = components.button(
            self.toolbar, 0, 1, "Save config", self.__save_config,
            tooltip="Save the current configuration as a custom named preset", width=90
        )
        self.wiki_btn = components.button(
            self.toolbar, 0, 2, "Wiki", self.open_wiki,
            tooltip="Opens the OneTrainer Wiki in your web browser, please give it a read!", width=50
        )
        self.wiki_btn.grid_configure(padx=(0, 8))

        # spacer inside toolbar to push right controls
        self.toolbar.grid_columnconfigure(3, weight=1)

        # right controls container (we will wrap its children)
        self.right_controls = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        self.right_controls.grid(row=0, column=4, sticky="e")

        # model type inside right_controls
        self.model_type_widget = components.options_kv(
            master=self.right_controls,
            row=0,
            column=0,
            values=[ #TODO simplify
                ("SD1.5", ModelType.STABLE_DIFFUSION_15),
                ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
                ("SD2.0", ModelType.STABLE_DIFFUSION_20),
                ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
                ("SD2.1", ModelType.STABLE_DIFFUSION_21),
                ("SD3", ModelType.STABLE_DIFFUSION_3),
                ("SD3.5", ModelType.STABLE_DIFFUSION_35),
                ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
                ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
                ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
                ("Stable Cascade", ModelType.STABLE_CASCADE_1),
                ("PixArt Alpha", ModelType.PIXART_ALPHA),
                ("PixArt Sigma", ModelType.PIXART_SIGMA),
                ("Flux Dev", ModelType.FLUX_DEV_1),
                ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
                ("Sana", ModelType.SANA),
                ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
                ("HiDream Full", ModelType.HI_DREAM_FULL),
                ("Chroma1", ModelType.CHROMA_1),
                ("QwenImage", ModelType.QWEN),
            ],
            ui_state=self.ui_state,
            var_name="model_type",
            command=self.__change_model_type,
        )

        # react to size changes (use toolbar width for wrapping decision)
        self.toolbar.bind('<Configure>', self._on_topbar_resize)

    def __create_training_method(self):
        if self.training_method:
            self.training_method.destroy()

        values = []
        #TODO simplify
        if self.train_config.model_type.is_stable_diffusion():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ]
        elif self.train_config.model_type.is_stable_diffusion_3() \
                or self.train_config.model_type.is_stable_diffusion_xl() \
                or self.train_config.model_type.is_wuerstchen() \
                or self.train_config.model_type.is_pixart() \
                or self.train_config.model_type.is_flux() \
                or self.train_config.model_type.is_sana() \
                or self.train_config.model_type.is_hunyuan_video() \
                or self.train_config.model_type.is_hi_dream() \
                or self.train_config.model_type.is_chroma():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]
        elif self.train_config.model_type.is_qwen():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
            ]

        # training method
        self.training_method = components.options_kv(
            master=self.right_controls,
            row=0,
            column=1,
            values=values,
            ui_state=self.ui_state,
            var_name="training_method",
            command=self.change_training_method_callback,
            width=105
        )
        self._update_wrap_state()

    def __change_model_type(self, model_type: ModelType):
        self.change_model_type_callback(model_type)
        self.__create_training_method()
        self._reposition_topbar_controls()

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.grid_forget()

        # place dropdown inside toolbar (row 0, col 0)
        self.configs_dropdown = components.options_kv(
            self.toolbar, 0, 0, self.configs, self.config_ui_state, "config_name", self.__load_current_config
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
            self.configs.sort()

    def __save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")

        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))

        return path

    def __save_secrets(self, path) -> str:
        write_json_atomic(path, self.train_config.secrets.to_dict())
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def __save_new_config(self, name):
        path = self.__save_to_file(name)

        is_new_config = name not in [x[0] for x in self.configs]

        if is_new_config:
            self.configs.append((name, path))
            self.configs.sort()

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

            with suppress(FileNotFoundError), open("secrets.json", "r") as f:
                secrets_dict=json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.train_config.from_dict(loaded_config.to_dict())
            self.ui_state.update(loaded_config)

            optimizer_config = change_optimizer(self.train_config)
            self.ui_state.get_var("optimizer").update(optimizer_config)

            self.load_preset_callback()
        except FileNotFoundError:
            pass
        except ctk.TclError:
            print(traceback.format_exc())

    def __remove_config(self):
        # TODO
        pass

    def save_default(self):
        self.__save_to_file("#")
        self.__save_secrets("secrets.json")

    def _on_topbar_resize(self, event):
        # Re-evaluate wrap on any toolbar size change (debounced)
        if not self._resize_pending:
            self._resize_pending = True
            self.toolbar.after_idle(self._update_wrap_state)

    def _update_wrap_state(self):
        """Decide wrapping based on measured widths of left-side controls and right_controls contents."""
        try:
            self._resize_pending = False
            self.toolbar.update_idletasks()
            tb = self.toolbar
            mt = getattr(self, 'model_type_widget', None)
            rc = getattr(self, 'right_controls', None)
            if not (mt and rc) or not tb.winfo_ismapped():
                return

            # total width needed for right controls (side-by-side); training_method may not exist yet
            tm = getattr(self, 'training_method', None)
            right_needed = mt.winfo_reqwidth()
            if tm and tm.winfo_exists():
                right_needed += tm.winfo_reqwidth()
            right_needed += 10  # small gap

            # width used by left side (dropdown + buttons)
            left_widgets = (
                getattr(self, 'configs_dropdown', None),
                getattr(self, 'save_btn', None),
                getattr(self, 'wiki_btn', None),
            )
            left_needed = sum(w.winfo_reqwidth() for w in left_widgets if w)

            total_width = max(1, tb.winfo_width())
            want_wrapped = (left_needed + right_needed) > total_width

            if want_wrapped != self._topbar_wrapped:
                self._topbar_wrapped = want_wrapped
                self._reposition_topbar_controls()
        except Exception:
            print(traceback.format_exc())

    def _reposition_topbar_controls(self):
        """Place right_controls either on row 0 (inline) or row 1 (below left side)."""
        try:
            if self._topbar_wrapped:
                # second row, left-aligned with fixed padding
                self.right_controls.grid_configure(
                    row=1,
                    column=0,
                    columnspan=5,
                    sticky="w",
                    padx=(8, 0),
                    pady=0,
                )
                self.model_type_widget.grid_configure(
                    row=0, column=0, padx=(0, 20), pady=0, sticky="w"
                )
                if self.training_method:
                    self.training_method.grid_configure(
                        row=0, column=1, padx=0, pady=0, sticky="w"
                    )
            else:
                # first row, right-aligned
                self.right_controls.grid_configure(row=0, column=4, columnspan=1, sticky="e", padx=0, pady=0)
                self.model_type_widget.grid_configure(row=0, column=0, padx=(0, 5), pady=0, sticky="e")
                if self.training_method:
                    self.training_method.grid_configure(row=0, column=1, padx=0, pady=0, sticky="e")
        except Exception:
            print(traceback.format_exc())
