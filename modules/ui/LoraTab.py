import copy

import customtkinter as ctk

from modules.modelSetup.PixArtAlphaLoRASetup import PRESETS as pixart_presets
from modules.modelSetup.StableDiffusionLoRASetup import PRESETS as sd_presets
from modules.modelSetup.StableDiffusionXLLoRASetup import PRESETS as sdxl_presets
from modules.modelSetup.WuerstchenLoRASetup import PRESETS as sc_presets
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class LoraTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(LoraTab, self).__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.layer_entry = None
        self.layer_selector = None
        self.presets = {}
        self.presets_list = []
        self.prior_custom = ""
        self.prior_selected = None
        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        if self.train_config.model_type.is_stable_diffusion():
            self.presets = sd_presets
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.presets = sdxl_presets
        elif self.train_config.model_type.is_wuerstchen():
            self.presets = sc_presets
        elif self.train_config.model_type.is_pixart():
            self.presets = pixart_presets
        else:
            self.presets = {"full": []}
        self.presets_list = list(self.presets.keys()) + ["custom"]
        self.setup_lora()

    def setup_lora(self):
        master = self.scroll_frame

        master.grid_columnconfigure(0, weight=0, uniform="a")
        master.grid_columnconfigure(1, weight=1, uniform="a")
        master.grid_columnconfigure(2, minsize=50, uniform="a")
        master.grid_columnconfigure(3, weight=0, uniform="a")
        master.grid_columnconfigure(4, weight=1, uniform="a")

        # lora model name
        components.label(master, 0, 0, "LoRA base model",
                         tooltip="The base LoRA to train on. Leave empty to create a new LoRA")
        components.file_entry(
            master, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # lora rank
        components.label(master, 1, 0, "LoRA rank",
                         tooltip="The rank parameter used when creating a new LoRA")
        components.entry(master, 1, 1, self.ui_state, "lora_rank")

        # lora rank
        components.label(master, 2, 0, "LoRA alpha",
                         tooltip="The alpha parameter used when creating a new LoRA")
        components.entry(master, 2, 1, self.ui_state, "lora_alpha")

        # Dropout Percentage
        components.label(master, 3, 0, "Dropout Probability",
                         tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
        components.entry(master, 3, 1, self.ui_state, "dropout_probability")

        # lora weight dtype
        components.label(master, 4, 0, "LoRA Weight Data Type",
                         tooltip="The LoRA weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 4, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        # For use with additional embeddings.
        components.label(master, 5, 0, "Bundle Embeddings",
                         tooltip="Bundles any additional embeddings into the LoRA output file, rather than as separate files")
        components.switch(master, 5, 1, self.ui_state, "bundle_additional_embeddings")

        components.label(master, 6, 0, "Layer Preset",
                         tooltip="Select a preset defining which layers to train, or select 'Custom' to define your own")
        self.layer_selector = components.options(
            master, 6, 1, self.presets_list, self.ui_state, "lora_layer_preset",
            command=self.__preset_set_layer_choice
        )

        self.layer_entry = components.entry(
            master, 6, 2, self.ui_state, "lora_layers",
            tooltip="Comma-separated list of diffusion layers to apply the lora to"
        )
        self.prior_custom = self.train_config.lora_layers or ""
        self.layer_entry.grid(row=6, column=2, columnspan=3, sticky="ew")
        # Some configs will come with the lora_layer_preset unset or wrong for
        # the new model, so let's set it now to a reasonable default so it hits
        # the UI correctly.
        v = self.ui_state.get_var("lora_layer_preset")
        if v.get() not in self.presets_list:
            v.set(self.presets_list[0])
        self.__preset_set_layer_choice(self.layer_selector.get())

    def __preset_set_layer_choice(self, selected: str):
        if not selected:
            selected = self.presets_list[0]

        if selected == "custom":
            self.layer_entry.configure(state="normal")
            self.layer_entry.cget('textvariable').set(self.prior_custom)
        else:
            if self.prior_selected == "custom":
                self.prior_custom = self.layer_entry.get()
            self.layer_entry.configure(state="readonly")
            self.layer_entry.cget('textvariable').set(",".join(self.presets[selected]))
        self.prior_selected = selected
