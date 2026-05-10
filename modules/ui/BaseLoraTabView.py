
from modules.util import path_util
from modules.util.enum.ModelType import PeftType
from modules.util.ui.validation_helpers import check_range


class BaseLoraTabView:
    def __init__(self, components):
        self.components = components

    def build(self, frame, controller, ui_state, setup_lora_callback):
        self.components.label(frame, 0, 0, "Type",
                              tooltip="The type of low-parameter finetuning method.")
        self.components.options_kv(frame, 0, 1, controller.get_peft_types(),
                                   ui_state, "peft_type", command=setup_lora_callback)

    def build_lora_options(self, master, controller, ui_state, peft_type: PeftType):
        if peft_type == PeftType.LOHA:
            name = "LoHa"
        elif peft_type == PeftType.OFT_2:
            name = "OFT v2"
        else:
            name = "LoRA"

        # lora model name
        self.components.label(master, 0, 0, f"{name} base model",
                              tooltip=f"The base {name} to train on. Leave empty to create a new {name}")
        self.components.path_entry(
            master, 0, 1, ui_state, "lora_model_name",
            mode="file", path_modifier=path_util.json_path_modifier,
            columnspan=4,
        )

        # LoRA decomposition
        if peft_type == PeftType.LORA:
            self.components.label(master, 1, 3, "Decompose Weights (DoRA)",
                                  tooltip="Decompose LoRA Weights (aka, DoRA).")
            self.components.switch(master, 1, 4, ui_state, "lora_decompose")

            self.components.label(master, 2, 3, "Use Norm Epsilon (DoRA Only)",
                                  tooltip="Add an epsilon to the norm divison calculation in DoRA. Can aid in training stability, and also acts as regularization.")
            self.components.switch(master, 2, 4, ui_state, "lora_decompose_norm_epsilon")
            self.components.label(master, 3, 3, "Apply on output axis (DoRA Only)",
                                  tooltip="Apply the weight decomposition on the output axis instead of the input axis.")
            self.components.switch(master, 3, 4, ui_state, "lora_decompose_output_axis")

        # LoRA and LoHA shared settings
        if peft_type == PeftType.LORA or peft_type == PeftType.LOHA:
            # rank
            self.components.label(master, 1, 0, f"{name} rank",
                                  tooltip=f"The rank parameter used when creating a new {name}")
            self.components.entry(master, 1, 1, ui_state, "lora_rank", required=True, extra_validate=check_range(lower=1, message="Rank must be at least 1"))

            # alpha
            self.components.label(master, 2, 0, f"{name} alpha",
                                  tooltip=f"The alpha parameter used when creating a new {name}")
            self.components.entry(master, 2, 1, ui_state, "lora_alpha", required=True)

            # Dropout Percentage
            self.components.label(master, 3, 0, "Dropout Probability",
                                  tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
            self.components.entry(master, 3, 1, ui_state, "dropout_probability")

            # weight dtype
            self.components.label(master, 4, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 4, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # For use with additional embeddings.
            self.components.label(master, 5, 0, "Bundle Embeddings",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 5, 1, ui_state, "bundle_additional_embeddings")

        # OFTv2
        elif peft_type == PeftType.OFT_2:
            # Block Size
            self.components.label(master, 1, 0, f"{name} Block Size",
                                  tooltip=f"The block size parameter used when creating a new {name}")
            self.components.entry(master, 1, 1, ui_state, "oft_block_size", required=True)

            # COFT
            self.components.label(master, 1, 3, "Constrained OFT (COFT)",
                                  tooltip="Use the constrained variant of OFT. This constrains the learned rotation to stay very close to the identity matrix, limiting adaptation to only small changes. This improves training stability, helps prevent overfitting on small datasets, and better preserves the base models original knowledge but it may lack expressiveness for tasks requiring substantial adaptation and introduces an additional hyperparameter (COFT Epsilon) that needs tuning.")
            self.components.switch(master, 1, 4, ui_state, "oft_coft")

            self.components.label(master, 2, 3, "COFT Epsilon",
                                  tooltip="The control strength of COFT. Only has an effect if COFT is enabled.")
            self.components.entry(master, 2, 4, ui_state, "coft_eps")

            # Block Share
            self.components.label(master, 3, 3, "Block Share",
                                  tooltip="Share the OFT parameters between blocks. A single rotation matrix is shared across all blocks within a layer, drastically cutting the number of trainable parameters and yielding very compact adapter files, potentially improving generalization but at the cost of significant expressiveness, which can lead to underfitting on more complex or diverse tasks.")
            self.components.switch(master, 3, 4, ui_state, "oft_block_share")

            # Dropout Percentage
            self.components.label(master, 2, 0, "Dropout Probability",
                                  tooltip="Dropout probability. This percentage of the rotated adapter nodes that will be randomly restored to the base model initial statue. Helps with overfitting. 0 disables, 1 maximum.")
            self.components.entry(master, 2, 1, ui_state, "dropout_probability")

            # OFT weight dtype
            self.components.label(master, 3, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 3, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # For use with additional embeddings.
            self.components.label(master, 4, 0, "Bundle Embeddings",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 4, 1, ui_state, "bundle_additional_embeddings")
