from pathlib import Path

from modules.modelSetup.PixArtAlphaLoRASetup import PRESETS as pixart_presets
from modules.modelSetup.StableDiffusion3LoRASetup import PRESETS as sd3_presets
from modules.modelSetup.StableDiffusionLoRASetup import PRESETS as sd_presets
from modules.modelSetup.StableDiffusionXLLoRASetup import PRESETS as sdxl_presets
from modules.modelSetup.WuerstchenLoRASetup import PRESETS as sc_presets
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class LoraTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

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
        self.options_frame = None

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
        elif self.train_config.model_type.is_stable_diffusion_3():
            self.presets = sd3_presets
        elif self.train_config.model_type.is_wuerstchen():
            self.presets = sc_presets
        elif self.train_config.model_type.is_pixart():
            self.presets = pixart_presets
        else:
            self.presets = {"full": []}
        self.presets_list = list(self.presets.keys()) + ["custom"]

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=2)

        components.label(self.scroll_frame, 0, 0, "Type",
                         tooltip="The type of low-parameter finetuning method.")
        # This will instantly call self.setup_lora.
        components.options_kv(self.scroll_frame, 0, 1, [
            ("LoRA", PeftType.LORA),
            ("LoHa", PeftType.LOHA),
        ], self.ui_state, "peft_type", command=self.setup_lora)

    def setup_lora(self, peft_type: PeftType):
        name = "LoHa" if peft_type == PeftType.LOHA else "LoRA"

        if self.options_frame:
            self.options_frame.destroy()
        self.options_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.options_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        master = self.options_frame

        master.grid_columnconfigure(0, weight=0, uniform="a")
        master.grid_columnconfigure(1, weight=1, uniform="a")
        master.grid_columnconfigure(2, minsize=50, uniform="a")
        master.grid_columnconfigure(3, weight=0, uniform="a")
        master.grid_columnconfigure(4, weight=1, uniform="a")

        # lora model name
        components.label(master, 0, 0, f"{name} base model",
                         tooltip=f"The base {name} to train on. Leave empty to create a new LoRA")
        entry = components.file_entry(
            master, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )
        entry.grid(row=0, column=1, columnspan=4)

        # lora rank
        components.label(master, 1, 0, f"{name} rank",
                         tooltip=f"The rank parameter used when creating a new {name}")
        components.entry(master, 1, 1, self.ui_state, "lora_rank")

        # decomposition
        if peft_type == PeftType.LORA:
            components.label(master, 1, 3, "Decompose Weights (DoRA)",
                             tooltip="Decompose LoRA Weights (aka, DoRA).")
            components.switch(master, 1, 4, self.ui_state, "lora_decompose")

            components.label(master, 2, 3, "Use Norm Espilon (DoRA Only)",
                             tooltip="Add an epsilon to the norm divison calculation in DoRA. Can aid in training stability, and also acts as regularization.")
            components.switch(master, 2, 4, self.ui_state, "lora_decompose_norm_epsilon")

        # lora rank
        components.label(master, 2, 0, f"{name} alpha",
                         tooltip="The alpha parameter used when creating a new f{name}")
        components.entry(master, 2, 1, self.ui_state, "lora_alpha")

        # Dropout Percentage
        components.label(master, 3, 0, "Dropout Probability",
                         tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
        components.entry(master, 3, 1, self.ui_state, "dropout_probability")

        # lora weight dtype
        components.label(master, 4, 0, f"{name} Weight Data Type",
                         tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 4, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        # For use with additional embeddings.
        components.label(master, 5, 0, "Bundle Embeddings",
                         tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
        components.switch(master, 5, 1, self.ui_state, "bundle_additional_embeddings")

        components.label(master, 6, 0, "Layer Preset",
                         tooltip="Select a preset defining which layers to train, or select 'Custom' to define your own")
        self.layer_selector = components.options(
            master, 6, 1, self.presets_list, self.ui_state, "lora_layer_preset",
            command=self.__preset_set_layer_choice
        )

        self.layer_entry = components.entry(
            master, 6, 2, self.ui_state, "lora_layers",
            tooltip=f"Comma-separated list of diffusion layers to apply the {name} to"
        )
        self.prior_custom = self.train_config.lora_layers or ""
        self.layer_entry.grid(row=6, column=2, columnspan=3, sticky="ew")
        # Some configs will come with the lora_layer_preset unset or wrong for
        # the new model, so let's set it now to a reasonable default so it hits
        # the UI correctly.
        if self.layer_selector.get() not in self.presets_list:
            self.layer_selector.set(self.presets_list[0])
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
