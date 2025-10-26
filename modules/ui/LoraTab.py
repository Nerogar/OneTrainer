from pathlib import Path

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

        self.scroll_frame = None
        self.options_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()
        self.scroll_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=0)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=2)

        components.label(self.scroll_frame, 0, 0, "Type",
                         tooltip="The type of low-parameter finetuning method.")
        # This will instantly call self.setup_lora.
        components.options_kv(self.scroll_frame, 0, 1, [
            ("LoRA", PeftType.LORA),
            ("LoHa", PeftType.LOHA),
            ("LoKr", PeftType.LOKR),
        ], self.ui_state, "peft_type", command=self.setup_lora)

    def setup_lora(self, peft_type: PeftType):
        if peft_type == PeftType.LOHA:
            name = "LoHa"
        elif peft_type == PeftType.LOKR:
            name = "LoKr"
        else:
            name = "LoRA"

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
                         tooltip=f"The base {name} to train on. Leave empty to create a new {name}")
        entry = components.file_entry(
            master, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )
        entry.grid(row=0, column=1, columnspan=4)

        # PEFT Type Specific Settings
        if peft_type == PeftType.LOKR:
            # LoKr Main Settings
            components.label(master, 1, 0, f"{name} dimension",
                             tooltip="The dimension parameter used for the secondary decomposition. Analogous to rank in LoRA.")
            components.entry(master, 1, 1, self.ui_state, "lokr_dim")

            components.label(master, 2, 0, "Decomposition Factor",
                             tooltip="Factor for Kronecker product decomposition. -1 for auto, which is recommended. Changing this drastically affects parameter count.")
            components.entry(master, 2, 1, self.ui_state, "lokr_decompose_factor")

            # LoKr Switches (Right side)
            components.label(master, 1, 3, "Decompose Both Matrices",
                             tooltip="Perform rank decomposition on both Kronecker product matrices (W1 and W2). Only effective for very small dimensions.")
            components.switch(master, 1, 4, self.ui_state, "lokr_decompose_both")

            components.label(master, 2, 3, "Use Tucker Decomposition (Conv)",
                             tooltip="Use Tucker decomposition for convolutional layers. Can be more efficient for some architectures.")
            components.switch(master, 2, 4, self.ui_state, "lokr_use_tucker")

            # LoKr DoRA Settings
            components.label(master, 4, 0, "Decompose Weights (DoRA)",
                             tooltip="Apply weight decomposition (DoRA) on top of the LoKr update.")
            components.switch(master, 4, 1, self.ui_state, "lokr_weight_decompose")

            components.label(master, 4, 3, "Apply DoRA on Output Axis",
                             tooltip="Apply the DoRA weight decomposition on the output axis instead of the input axis.")
            components.switch(master, 4, 4, self.ui_state, "lokr_dora_on_output")

            # LoKr Dropout Settings
            components.label(master, 5, 0, "Rank Dropout",
                             tooltip="Dropout probability for the rank dimension. Helps with overfitting.")
            components.entry(master, 5, 1, self.ui_state, "lokr_rank_dropout")

            components.label(master, 5, 3, "Module Dropout",
                             tooltip="Dropout probability for the entire LoKr module. A percentage of modules will be skipped during training steps.")
            components.entry(master, 5, 4, self.ui_state, "lokr_module_dropout")

            components.label(master, 6, 0, "Scale Rank Dropout",
                             tooltip="If using Rank Dropout, scale the remaining weights to maintain variance.")
            components.switch(master, 6, 1, self.ui_state, "lokr_rank_dropout_scale")

            # LoKr Advanced/Expert Settings
            components.label(master, 7, 0, "Use Scalar",
                             tooltip="Use a learnable scalar parameter. Can sometimes improve performance.")
            components.switch(master, 7, 1, self.ui_state, "lokr_use_scalar")

            components.label(master, 7, 3, "Unbalanced Factorization",
                             tooltip="Use an unbalanced factorization for dimensions. Experimental.")
            components.switch(master, 7, 4, self.ui_state, "lokr_unbalanced_factorization")

            components.label(master, 8, 0, "Force Full Matrix (W2)",
                             tooltip="Forces the second Kronecker matrix (W2) to be a full matrix, ignoring the dimension/rank setting. For expert use.")
            components.switch(master, 8, 1, self.ui_state, "lokr_full_matrix")

            components.label(master, 8, 3, "Use RS-LoKr Scaling",
                             tooltip="Root-Squared LoKr scaling. Changes the scaling factor from alpha/dim to alpha/sqrt(dim). Experimental.")
            components.switch(master, 8, 4, self.ui_state, "lokr_rs_lora")


        elif peft_type == PeftType.LORA:
            components.label(master, 1, 0, f"{name} rank",
                             tooltip=f"The rank parameter used when creating a new {name}")
            components.entry(master, 1, 1, self.ui_state, "lora_rank")

            components.label(master, 2, 3, "Decompose Weights (DoRA)",
                             tooltip="Decompose LoRA Weights (aka, DoRA).")
            components.switch(master, 2, 4, self.ui_state, "lora_decompose")

            components.label(master, 3, 3, "Use Norm Epsilon (DoRA Only)",
                             tooltip="Add an epsilon to the norm divison calculation in DoRA. Can aid in training stability, and also acts as regularization.")
            components.switch(master, 3, 4, self.ui_state, "lora_decompose_norm_epsilon")
            components.label(master, 3, 3, "Apply on output axis (DoRA Only)",
                             tooltip="Apply the weight decomposition on the output axis instead of the input axis.")
            components.switch(master, 3, 4, self.ui_state, "lora_decompose_output_axis")

        else: # LoHa
            components.label(master, 1, 0, f"{name} rank",
                             tooltip=f"The rank parameter used when creating a new {name}")
            components.entry(master, 1, 1, self.ui_state, "lora_rank")


        # Shared Settings (Bottom)
        # Use a dynamic row index to place these controls after the specific ones
        next_row = master.grid_size()[1]

        components.label(master, next_row, 0, f"{name} alpha",
                         tooltip=f"The alpha parameter used when creating a new {name}")
        components.entry(master, next_row, 1, self.ui_state, "lora_alpha")

        if peft_type != PeftType.LOKR:
            components.label(master, next_row + 1, 0, "Dropout Probability",
                            tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
            components.entry(master, next_row + 1, 1, self.ui_state, "dropout_probability")

        components.label(master, next_row + 2, 0, f"{name} Weight Data Type",
                         tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, next_row + 2, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        components.label(master, next_row + 3, 0, "Bundle Embeddings",
                         tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
        components.switch(master, next_row + 3, 1, self.ui_state, "bundle_additional_embeddings")
