
from modules.util import path_util
from modules.util.enum.ModelType import PeftType
from modules.util.ui.validation_helpers import check_range


class BaseLoraTabView:
    def __init__(self, components):
        self.components = components

    def build(self, frame, controller, ui_state, setup_lora_callback):
        self.components.label(frame, 0, 0, "类型",
                              tooltip="低参数微调方法类型")
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
                                  tooltip="分解LoRA权重（即DoRA）")
            self.components.switch(master, 1, 4, ui_state, "lora_decompose")

            self.components.label(master, 2, 3, "Use Norm Epsilon (DoRA Only)",
                                  tooltip="在DoRA范数除法中添加epsilon，有助于训练稳定性")
            self.components.switch(master, 2, 4, ui_state, "lora_decompose_norm_epsilon")
            self.components.label(master, 3, 3, "在输出轴应用（仅DoRA）",
                                  tooltip="在输出轴而非输入轴应用权重分解")
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
            self.components.label(master, 3, 0, "丢弃概率",
                                  tooltip="丢弃概率，每步随机忽略此比例的模型节点，0=禁用")
            self.components.entry(master, 3, 1, ui_state, "dropout_probability")

            # weight dtype
            self.components.label(master, 4, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 4, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # For use with additional embeddings.
            self.components.label(master, 5, 0, "捆绑嵌入",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 5, 1, ui_state, "bundle_additional_embeddings")

        # OFTv2
        elif peft_type == PeftType.OFT_2:
            # Block Size
            self.components.label(master, 1, 0, f"{name} Block Size",
                                  tooltip=f"The block size parameter used when creating a new {name}")
            self.components.entry(master, 1, 1, ui_state, "oft_block_size", required=True)

            # Block Share
            self.components.label(master, 1, 3, "块共享",
                                  tooltip="块间共享OFT参数，大幅减少可训练参数，但可能降低表达能力")
            self.components.switch(master, 1, 4, ui_state, "oft_block_share")

            # Scaled OFT (SOFT)
            self.components.label(master, 2, 3, "Scaled OFT (SOFT)",
                                  tooltip="学习权重缩放因子，确保不同块大小下有效学习率一致")
            self.components.switch(master, 2, 4, ui_state, "oft_scaled")

            # Dropout Percentage
            self.components.label(master, 2, 0, "丢弃概率",
                                  tooltip="Dropout probability. This percentage of the rotated adapter nodes that will be randomly restored to the base model initial statue. Helps with overfitting. 0 disables, 1 maximum.")
            self.components.entry(master, 2, 1, ui_state, "dropout_probability")

            # OFT weight dtype
            self.components.label(master, 3, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 3, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # For use with additional embeddings.
            self.components.label(master, 4, 0, "捆绑嵌入",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 4, 1, ui_state, "bundle_additional_embeddings")

        # LoKr
        elif peft_type == PeftType.LOKR:
            # LoKr Main Settings
            self.components.label(master, 1, 0, f"{name} dimension",
                                  tooltip="二次分解的维度参数，类似于LoRA的秩")
            self.components.entry(master, 1, 1, ui_state, "lokr_dim")

            self.components.label(master, 2, 0, "分解因子",
                                  tooltip="Kronecker积分解因子，-1为自动（推荐）")
            self.components.entry(master, 2, 1, ui_state, "lokr_decompose_factor")

            # alpha
            self.components.label(master, 3, 0, f"{name} alpha",
                                  tooltip=f"The alpha parameter used when creating a new {name}")
            self.components.entry(master, 3, 1, ui_state, "lora_alpha")

            # Dropout Percentage
            self.components.label(master, 4, 0, "丢弃概率",
                                  tooltip="丢弃概率，每步随机忽略此比例的模型节点，0=禁用")
            self.components.entry(master, 4, 1, ui_state, "dropout_probability")

            # LoKr weight dtype
            self.components.label(master, 5, 0, f"{name} Weight Data Type",
                                  tooltip=f"The {name} weight data type used for training. This can reduce memory consumption, but reduces precision")
            self.components.options_kv(master, 5, 1, controller.get_lora_weight_dtypes(), ui_state, "lora_weight_dtype")

            # LoKr Vectorization trick
            self.components.label(master, 6, 0, "Kronecker-Vec Trick",
                                  tooltip="使用加速路径绕过完整Kronecker积的实现，大幅加速LoKr")
            self.components.switch(master, 6, 1, ui_state, "lokr_vec_trick")

            # LoKr Decomposition Settings
            self.components.label(master, 1, 3, "分解两个矩阵",
                                  tooltip="对两个Kronecker积矩阵进行秩分解，仅对极小维度有效")
            self.components.switch(master, 1, 4, ui_state, "lokr_decompose_both")

            self.components.label(master, 2, 3, "Use Tucker Decomposition (Conv)",
                                  tooltip="对卷积层使用Tucker分解，某些架构更高效")
            self.components.switch(master, 2, 4, ui_state, "lokr_use_tucker")

            self.components.label(master, 3, 3, "Force Full Matrix (W2)",
                                  tooltip="强制第二个Kronecker矩阵为全矩阵，忽略维度设置")
            self.components.switch(master, 3, 4, ui_state, "lokr_full_matrix")

            # LoKr DoRA Settings
            self.components.label(master, 4, 3, "Decompose Weights (DoRA)",
                                  tooltip="在LoKr更新上应用权重分解（DoRA）")
            self.components.switch(master, 4, 4, ui_state, "lokr_weight_decompose")

            self.components.label(master, 5, 3, "在输出轴应用DoRA",
                                  tooltip="在输出轴而非输入轴应用DoRA权重分解")
            self.components.switch(master, 5, 4, ui_state, "lokr_dora_on_output")

            # Additional embeddings
            self.components.label(master, 6, 3, "捆绑嵌入",
                                  tooltip=f"Bundles any additional embeddings into the {name} output file, rather than as separate files")
            self.components.switch(master, 6, 4, ui_state, "bundle_additional_embeddings")
