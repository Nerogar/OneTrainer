
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
        elif peft_type == PeftType.LOKR:
            name = "LoKr"
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

            # Block Share
            self.components.label(master, 1, 3, "Block Share",
                                  tooltip="Share the OFT parameters between blocks. A single rotation matrix is shared across all blocks within a layer, drastically cutting the number of trainable parameters and yielding very compact adapter files, potentially improving generalization but at the cost of significant expressiveness, which can lead to underfitting on more complex or diverse tasks.")
            self.components.switch(master, 1, 4, ui_state, "oft_block_share")

            # Scaled OFT (SOFT)
            self.components.label(master, 2, 3, "Scaled OFT (SOFT)",
                                  tooltip="Applies a scaling factor to the learned weights. This ensures that the effective learning rate remains consistent across different block sizes. Without this, different block sizes require significantly different learning rates.")
            self.components.switch(master, 2, 4, ui_state, "oft_scaled")

            # DoRA-OFT (DOFT)
            self.components.label(master, 3, 3, "DoRA OFT (DOFT)",
                                  tooltip="Combines Weight-Decomposed Low-Rank Adaptation (DoRA) with OFT. By decoupling the weight into magnitude and direction components, it achieves the superior training dynamics of DoRA but with the stability and performance of OFT. Because OFT is norm-preserving, it avoids the heavy re-calculations typically found in standard DoRA, resulting in faster training (same speed as standard OFT) and better convergence.")
            self.components.switch(master, 3, 4, ui_state, "dora_oft")

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

        # LoKr
        elif peft_type == PeftType.LOKR:
            # LoKr Main Settings
            self.components.label(master, 1, 0, f"{name} dimension",
                                  tooltip="The dimension parameter used for the secondary decomposition. Analogous to rank in LoRA.")
            self.components.entry(master, 1, 1, ui_state, "lokr_dim")

            self.components.label(master, 2, 0, "Decomposition Factor",
                                  tooltip="Factor for Kronecker product decomposition. -1 for auto, which is recommended. Changing this drastically affects parameter count.")
            self.components.entry(master, 2, 1, ui_state, "lokr_decompose_factor")

            # alpha
            self.components.label(master, 3, 0, f"{name} alpha",
                                  tooltip=f"The alpha parameter used when creating a new {name}")
            self.components.entry(master, 3, 1, ui_state, "lora_alpha")

            # Dropout Percentage
            self.components.label(master, 4, 0, "Dropout Probability",
                                  tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
            self.components.entry(master, 4, 1, ui_state, "dropout_probability")

            # LoKr weight dtype
            self.components.label(master, 5, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 5, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # LoKr Vectorization trick
            self.components.label(master, 6, 0, "Kronecker-Vec Trick",
                                  tooltip="Uses an accelerated path that bypasses the materialization of the full Kronecker product. This delivers a massive speedup to the LoKr without sacrificing precision. Highly recommended.")
            self.components.switch(master, 6, 1, ui_state, "lokr_vec_trick")

            # LoKr Decomposition Settings
            self.components.label(master, 1, 3, "Decompose Both Matrices",
                                  tooltip="Perform rank decomposition on both Kronecker product matrices (W1 and W2). Only effective for very small dimensions.")
            self.components.switch(master, 1, 4, ui_state, "lokr_decompose_both")

            self.components.label(master, 2, 3, "Use Tucker Decomposition (Conv)",
                                  tooltip="Use Tucker decomposition for convolutional layers. Can be more efficient for some architectures.")
            self.components.switch(master, 2, 4, ui_state, "lokr_use_tucker")

            self.components.label(master, 3, 3, "Force Full Matrix (W2)",
                                  tooltip="Forces the second Kronecker matrix (W2) to be a full matrix, ignoring the dimension setting. For expert use.")
            self.components.switch(master, 3, 4, ui_state, "lokr_full_matrix")

            # LoKr DoRA Settings
            self.components.label(master, 4, 3, "Decompose Weights (DoRA)",
                                  tooltip="Apply weight decomposition (DoRA) on top of the LoKr update.")
            self.components.switch(master, 4, 4, ui_state, "lokr_weight_decompose")

            self.components.label(master, 5, 3, "Apply DoRA on Output Axis",
                                  tooltip="Apply the DoRA weight decomposition on the output axis instead of the input axis.")
            self.components.switch(master, 5, 4, ui_state, "lokr_dora_on_output")

            # Additional embeddings
            self.components.label(master, 6, 3, "Bundle Embeddings",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 6, 4, ui_state, "bundle_additional_embeddings")
