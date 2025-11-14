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
            ("OFT v2", PeftType.OFT_2),
        ], self.ui_state, "peft_type", command=self.setup_lora)

    def setup_lora(self, peft_type: PeftType):
        if peft_type == PeftType.LOHA:
            name = "LoHa"
        elif peft_type == PeftType.OFT_2:
            name = "OFT v2"
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


        # LoRA decomposition
        if peft_type == PeftType.LORA:
            components.label(master, 1, 3, "Decompose Weights (DoRA)",
                             tooltip="Decompose LoRA Weights (aka, DoRA).")
            components.switch(master, 1, 4, self.ui_state, "lora_decompose")

            components.label(master, 2, 3, "Use Norm Epsilon (DoRA Only)",
                             tooltip="Add an epsilon to the norm divison calculation in DoRA. Can aid in training stability, and also acts as regularization.")
            components.switch(master, 2, 4, self.ui_state, "lora_decompose_norm_epsilon")
            components.label(master, 3, 3, "Apply on output axis (DoRA Only)",
                             tooltip="Apply the weight decomposition on the output axis instead of the input axis.")
            components.switch(master, 3, 4, self.ui_state, "lora_decompose_output_axis")

        # LoRA and LoHA shared settings
        if peft_type == PeftType.LORA or peft_type == PeftType.LOHA:
            # rank
            components.label(master, 1, 0, f"{name} rank",
                            tooltip=f"The rank parameter used when creating a new {name}")
            components.entry(master, 1, 1, self.ui_state, "lora_rank")

            # alpha
            components.label(master, 2, 0, f"{name} alpha",
                            tooltip=f"The alpha parameter used when creating a new {name}")
            components.entry(master, 2, 1, self.ui_state, "lora_alpha")

            # Dropout Percentage
            components.label(master, 3, 0, "Dropout Probability",
                            tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
            components.entry(master, 3, 1, self.ui_state, "dropout_probability")

            # weight dtype
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

        # OFTv2
        elif peft_type == PeftType.OFT_2:
            # Block Size
            components.label(master, 1, 0, f"{name} Block Size",
                            tooltip=f"The block size parameter used when creating a new {name}")
            components.entry(master, 1, 1, self.ui_state, "oft_block_size")

            # COFT
            components.label(master, 1, 3, "Constrained OFT (COFT)",
                             tooltip="Use the constrained variant of OFT. This constrains the learned rotation to stay very close to the identity matrix, limiting adaptation to only small changes. This improves training stability, helps prevent overfitting on small datasets, and better preserves the base models original knowledge but it may lack expressiveness for tasks requiring substantial adaptation and introduces an additional hyperparameter (COFT Epsilon) that needs tuning.")
            components.switch(master, 1, 4, self.ui_state, "oft_coft")

            components.label(master, 2, 3, "COFT Epsilon",
                             tooltip="The control strength of COFT. Only has an effect if COFT is enabled.")
            components.entry(master, 2, 4, self.ui_state, "coft_eps")

            # Block Share
            components.label(master, 3, 3, "Block Share",
                             tooltip="Share the OFT parameters between blocks. A single rotation matrix is shared across all blocks within a layer, drastically cutting the number of trainable parameters and yielding very compact adapter files, potentially improving generalization but at the cost of significant expressiveness, which can lead to underfitting on more complex or diverse tasks.")
            components.switch(master, 3, 4, self.ui_state, "oft_block_share")

            # Dropout Percentage
            components.label(master, 2, 0, "Dropout Probability",
                            tooltip="Dropout probability. This percentage of the rotated adapter nodes that will be randomly restored to the base model initial statue. Helps with overfitting. 0 disables, 1 maximum.")
            components.entry(master, 2, 1, self.ui_state, "dropout_probability")

            # OFT weight dtype
            components.label(master, 3, 0, f"{name} Weight Data Type",
                            tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            components.options_kv(master, 3, 1, [
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
            ], self.ui_state, "lora_weight_dtype")

            # For use with additional embeddings.
            components.label(master, 4, 0, "Bundle Embeddings",
                            tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            components.switch(master, 4, 1, self.ui_state, "bundle_additional_embeddings")
