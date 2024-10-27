from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
from modules.ui.SchedulerParamsWindow import SchedulerParamsWindow
from modules.ui.TimestepDistributionWindow import TimestepDistributionWindow
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.optimizer_util import change_optimizer
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class TrainingTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=1)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=1)

        column_0 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_0.grid(row=0, column=0, sticky="nsew")
        column_0.grid_columnconfigure(0, weight=1)

        column_1 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_1.grid(row=0, column=1, sticky="nsew")
        column_1.grid_columnconfigure(0, weight=1)

        column_2 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_2.grid(row=0, column=2, sticky="nsew")
        column_2.grid_columnconfigure(0, weight=1)

        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui(column_0, column_1, column_2)
        if self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui(column_0, column_1, column_2)

    def __setup_stable_diffusion_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)

    def __setup_stable_diffusion_3_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_1_frame(column_0, 1, supports_include=True)
        self.__create_text_encoder_2_frame(column_0, 2, supports_include=True)
        self.__create_text_encoder_3_frame(column_0, 3, supports_include=True)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)

    def __setup_stable_diffusion_xl_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_1_frame(column_0, 1)
        self.__create_text_encoder_2_frame(column_0, 2)
        self.__create_embedding_frame(column_0, 3)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)

    def __setup_wuerstchen_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_prior_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 0)
        self.__create_loss_frame(column_2, 1)

    def __setup_pixart_alpha_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_prior_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2, supports_vb_loss=True)

    def __setup_flux_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_1_frame(column_0, 1, supports_include=True)
        self.__create_text_encoder_2_frame(column_0, 2, supports_include=True)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)

    def __create_base_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # optimizer
        components.label(frame, 0, 0, "优化器",
                         tooltip="优化器类型")
        components.options_adv(frame, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer.optimizer",
                               command=self.__restore_optimizer_config, adv_command=self.__open_optimizer_params_window)

        # learning rate scheduler
        # Wackiness will ensue when reloading configs if we don't check and clear this first.
        if hasattr(self, "lr_scheduler_comp"):
            delattr(self, "lr_scheduler_comp")
            delattr(self, "lr_scheduler_adv_comp")
        components.label(frame, 1, 0, "学习率调度器",
                         tooltip="学习率调度器，在训练期间自动更改学习率")
        _, d = components.options_adv(frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                                      "learning_rate_scheduler", command=self.__restore_scheduler_config,
                                      adv_command=self.__open_scheduler_params_window)
        self.lr_scheduler_comp = d['component']
        self.lr_scheduler_adv_comp = d['button_component']
        # Initial call requires the presence of self.lr_scheduler_adv_comp.
        self.__restore_scheduler_config(self.ui_state.get_var("learning_rate_scheduler").get())

        # learning rate
        components.label(frame, 2, 0, "学习率",
                         tooltip="基础学习率")
        components.entry(frame, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(frame, 3, 0, "学习率预热步数",
                         tooltip="将学习率从 0 逐渐增加到指定学习率所需的步数")
        components.entry(frame, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate cycles
        components.label(frame, 4, 0, "学习率周期",
                         tooltip="学习率周期的数量。这仅适用于学习率调度器支持周期的情况")
        components.entry(frame, 4, 1, self.ui_state, "learning_rate_cycles")

        # epochs
        components.label(frame, 5, 0, "轮次",
                         tooltip="完整训练运行的轮次数量")
        components.entry(frame, 5, 1, self.ui_state, "epochs")

        # batch size
        components.label(frame, 6, 0, "批次大小",
                         tooltip="一个训练步骤的批次大小")
        components.entry(frame, 6, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(frame, 7, 0, "累积步数",
                         tooltip="累积步数。增加此数字可以以训练速度为代价换取批次大小")
        components.entry(frame, 7, 1, self.ui_state, "gradient_accumulation_steps")

        # Learning Rate Scaler
        components.label(frame, 8, 0, "学习率缩放器",
                         tooltip="选择在训练期间使用的学习率缩放类型。功能上等同于：LR * SQRT(选择)")
        components.options(frame, 8, 1, [str(x) for x in list(LearningRateScaler)], self.ui_state,
                           "learning_rate_scaler")

    def __create_base2_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # attention mechanism
        components.label(frame, 0, 0, "注意力",
                         tooltip="训练期间使用的注意力机制。这会对速度和内存消耗产生重大影响")
        components.options(frame, 0, 1, [str(x) for x in list(AttentionMechanism)], self.ui_state,
                           "attention_mechanism")

        # ema
        components.label(frame, 1, 0, "EMA",
                         tooltip="EMA 对许多步骤的训练进度进行平均，更好地保留大型数据集中的不同概念")
        components.options(frame, 1, 1, [str(x) for x in list(EMAMode)], self.ui_state,
                           "ema")

        # ema decay
        components.label(frame, 2, 0, "EMA 衰减",
                         tooltip="EMA 模型的衰减参数。较高的数字将平均更多步骤。对于数百或数千张图像的数据集，将其设置为 0.9999。对于较小的数据集，将其设置为 0.999 或 0.998")
        components.entry(frame, 2, 1, self.ui_state, "ema_decay")

        # ema update step interval
        components.label(frame, 3, 0, "EMA 更新步长间隔",
                         tooltip="EMA 更新步长之间的步长数")
        components.entry(frame, 3, 1, self.ui_state, "ema_update_step_interval")

        # gradient checkpointing
        components.label(frame, 4, 0, "梯度检查点",
                         tooltip="启用梯度检查点。这会减少内存使用量，但会增加训练时间")
        components.options(frame, 4, 1, [str(x) for x in list(GradientCheckpointingMethod)], self.ui_state,
                           "gradient_checkpointing")

        # train dtype
        components.label(frame, 5, 0, "训练数据类型",
                         tooltip="用于训练的混合精度数据类型。这可以提高训练速度，但会降低精度")
        components.options_kv(frame, 5, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")

        # fallback train dtype
        components.label(frame, 6, 0, "回退训练数据类型",
                         tooltip="用于不支持 float16 数据类型的训练阶段的混合精度数据类型。这可以提高训练速度，但会降低精度")
        components.options_kv(frame, 6, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "fallback_train_dtype")

        # autocast cache
        components.label(frame, 7, 0, "自动广播缓存",
                         tooltip="启用自动广播缓存。禁用此选项会减少内存使用量，但会增加训练时间")
        components.switch(frame, 7, 1, self.ui_state, "enable_autocast_cache")

        # resolution
        components.label(frame, 8, 0, "分辨率",
                         tooltip="用于训练的分辨率。可以选择指定多个分辨率（用逗号分隔），或以 <宽度>x<高度> 的格式指定单个精确分辨率")
        components.entry(frame, 8, 1, self.ui_state, "resolution")

        # force circular padding
        components.label(frame, 9, 0, "强制循环填充",
                         tooltip="为所有卷积层启用循环填充，以便更好地训练无缝图像")
        components.switch(frame, 9, 1, self.ui_state, "force_circular_padding")

    def __create_align_prop_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # align prop
        components.label(frame, 0, 0, "AlignProp",
                         tooltip="启用 AlignProp 训练\n以奖励反向传播对齐文本到图像扩散模型")
        components.switch(frame, 0, 1, self.ui_state, "align_prop")

        # align prop probability
        components.label(frame, 1, 0, "AlignProp 概率",
                         tooltip="当启用 AlignProp 时，指定使用 AlignProp 计算进行的训练步骤数量")
        components.entry(frame, 1, 1, self.ui_state, "align_prop_probability")

        # align prop loss
        components.label(frame, 2, 0, "AlignProp 损失",
                         tooltip="指定用于 AlignProp 计算的损失函数")
        components.options(frame, 2, 1, [str(x) for x in list(AlignPropLoss)], self.ui_state, "align_prop_loss")

        # align prop weight
        components.label(frame, 3, 0, "AlignProp 权重",
                         tooltip="AlignProp 损失的权重乘数")
        components.entry(frame, 3, 1, self.ui_state, "align_prop_weight")

        # align prop steps
        components.label(frame, 4, 0, "AlignProp 步长",
                         tooltip="每个 AlignProp 步长的推理步长数")
        components.entry(frame, 4, 1, self.ui_state, "align_prop_steps")

        # align prop truncate steps
        components.label(frame, 5, 0, "AlignProp 截断步长",
                         tooltip="使用 AlignProp 时随机截断的步长比例。这需要增加模型的多样性。")
        components.entry(frame, 5, 1, self.ui_state, "align_prop_truncate_steps")

        # align prop truncate steps
        components.label(frame, 6, 0, "AlignProp CFG 缩放",
                         tooltip="AlignProp 计算推理步长的 CFG 缩放")
        components.entry(frame, 6, 1, self.ui_state, "align_prop_cfg_scale")

    def __create_text_encoder_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train text encoder
        components.label(frame, 0, 0, "训练文本编码器",
                         tooltip="启用训练文本编码器模型")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder.train")

        # dropout
        components.label(frame, row, 0, "丢弃概率",
                         tooltip="文本编码器条件丢弃的概率")
        components.entry(frame, row, 1, self.ui_state, "text_encoder.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, 2, 0, "停止训练时间",
                         tooltip="何时停止训练文本编码器")
        components.time_entry(frame, 2, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 3, 0, "文本编码器学习率",
                         tooltip="文本编码器的学习率。覆盖基础学习率")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder.learning_rate")

        # text encoder layer skip (clip skip)
        components.label(frame, 4, 0, "剪辑跳过",
                         tooltip="要跳过的额外剪辑层数。0 = 模型默认值")
        components.entry(frame, 4, 1, self.ui_state, "text_encoder_layer_skip")

    def __create_text_encoder_1_frame(self, master, row, supports_include: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "包含文本编码器 1",
                             tooltip="在训练运行中包含文本编码器 1")
            components.switch(frame, row, 1, self.ui_state, "text_encoder.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "训练文本编码器 1",
                         tooltip="启用训练文本编码器 1 模型")
        components.switch(frame, row, 1, self.ui_state, "text_encoder.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "训练文本编码器 1 嵌入",
                         tooltip="启用训练文本编码器 1 模型的嵌入")
        components.switch(frame, row, 1, self.ui_state, "text_encoder.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "丢弃概率",
                         tooltip="文本编码器 1 条件丢弃的概率")
        components.entry(frame, row, 1, self.ui_state, "text_encoder.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "停止训练时间",
                         tooltip="何时停止训练文本编码器 1")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "文本编码器 1 学习率",
                         tooltip="文本编码器 1 的学习率。覆盖基础学习率")
        components.entry(frame, row, 1, self.ui_state, "text_encoder.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "文本编码器 1 剪辑跳过",
                         tooltip="要跳过的额外剪辑层数。0 = 模型默认值")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_layer_skip")
        row += 1

    def __create_text_encoder_2_frame(self, master, row, supports_include: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "Include Text Encoder 2",
                             tooltip="Includes text encoder 2 in the training run")
            components.switch(frame, row, 1, self.ui_state, "text_encoder_2.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "Train Text Encoder 2",
                         tooltip="Enables training the text encoder 2 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_2.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "Train Text Encoder 2 Embedding",
                         tooltip="Enables training embeddings for the text encoder 2 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_2.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder 2 conditioning")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 2")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder_2.stop_training_after",
                              "text_encoder_2.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "Text Encoder 2 Learning Rate",
                         tooltip="The learning rate of the text encoder 2. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "Text Encoder 2 Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2_layer_skip")
        row += 1

    def __create_text_encoder_3_frame(self, master, row, supports_include: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "Include Text Encoder 3",
                             tooltip="Includes text encoder 3 in the training run")
            components.switch(frame, row, 1, self.ui_state, "text_encoder_3.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "Train Text Encoder 3",
                         tooltip="Enables training the text encoder 3 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_3.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "Train Text Encoder 3 Embedding",
                         tooltip="Enables training embeddings for the text encoder 3 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_3.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder 3 conditioning")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 3")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder_3.stop_training_after",
                              "text_encoder_3.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "Text Encoder 3 Learning Rate",
                         tooltip="The learning rate of the text encoder 3. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "Text Encoder 3 Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3_layer_skip")
        row += 1

    def __create_embedding_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")

        # embedding learning rate
        components.label(frame, 0, 0, "嵌入学习率",
                         tooltip="嵌入的学习率。覆盖基础学习率")
        components.entry(frame, 0, 1, self.ui_state, "embedding_learning_rate")

        # preserve embedding norm
        components.label(frame, 1, 0, "保留嵌入范数",
                         tooltip="将每个训练后的嵌入重新缩放到中位嵌入范数")
        components.switch(frame, 1, 1, self.ui_state, "preserve_embedding_norm")

    def __create_unet_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train unet
        components.label(frame, 0, 0, "训练 UNet",
                         tooltip="启用训练 UNet 模型")
        components.switch(frame, 0, 1, self.ui_state, "unet.train")

        # train unet epochs
        components.label(frame, 1, 0, "停止训练时间",
                         tooltip="何时停止训练 UNet")
        components.time_entry(frame, 1, 1, self.ui_state, "unet.stop_training_after", "unet.stop_training_after_unit",
                              supports_time_units=False)

        # unet learning rate
        components.label(frame, 2, 0, "UNet 学习率",
                         tooltip="UNet 的学习率。覆盖基础学习率")
        components.entry(frame, 2, 1, self.ui_state, "unet.learning_rate")

        # rescale noise scheduler to zero terminal SNR
        components.label(frame, 3, 0, "重新缩放到零终端信噪比",
                         tooltip="将噪声调度器重新缩放到零终端信噪比，并将模型切换到 v 预测目标")
        components.switch(frame, 3, 1, self.ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")

    def __create_prior_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train prior
        components.label(frame, 0, 0, "训练先验",
                         tooltip="启用训练先验模型")
        components.switch(frame, 0, 1, self.ui_state, "prior.train")

        # train prior epochs
        components.label(frame, 1, 0, "停止训练时间",
                         tooltip="何时停止训练先验")
        components.time_entry(frame, 1, 1, self.ui_state, "prior.stop_training_after", "prior.stop_training_after_unit",
                              supports_time_units=False)

        # prior learning rate
        components.label(frame, 2, 0, "先验学习率",
                         tooltip="先验的学习率。覆盖基础学习率")
        components.entry(frame, 2, 1, self.ui_state, "prior.learning_rate")

    def __create_transformer_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train transformer
        components.label(frame, 0, 0, "训练 Transformer",
                         tooltip="启用训练 Transformer 模型")
        components.switch(frame, 0, 1, self.ui_state, "prior.train")

        # train transformer epochs
        components.label(frame, 1, 0, "停止训练时间",
                         tooltip="何时停止训练 Transformer")
        components.time_entry(frame, 1, 1, self.ui_state, "prior.stop_training_after", "prior.stop_training_after_unit",
                              supports_time_units=False)

        # transformer learning rate
        components.label(frame, 2, 0, "Transformer 学习率",
                         tooltip="Transformer 的学习率。覆盖基础学习率")
        components.entry(frame, 2, 1, self.ui_state, "prior.learning_rate")

        # transformer learning rate
        components.label(frame, 3, 0, "强制注意力掩码",
                         tooltip="强制启用将文本嵌入注意力掩码传递到 Transformer。这可以改善对较短标题的训练。")
        components.switch(frame, 3, 1, self.ui_state, "prior.attention_mask")

    def __create_noise_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # offset noise weight
        components.label(frame, 0, 0, "Offset噪声权重",
                         tooltip="添加到每个训练步骤的offset噪声的权重")
        components.entry(frame, 0, 1, self.ui_state, "offset_noise_weight")

        # perturbation noise weight
        components.label(frame, 1, 0, "Perturbation噪声权重",
                         tooltip="添加到每个训练步骤的Perturbation噪声的权重")
        components.entry(frame, 1, 1, self.ui_state, "perturbation_noise_weight")

        # timestep distribution
        components.label(frame, 2, 0, "时间步长分布",
                         tooltip="选择在训练期间采样时间步长的函数",
                         wide_tooltip=True)
        components.options_adv(frame, 2, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state, "timestep_distribution",
                               adv_command=self.__open_timestep_distribution_window)

        # min noising strength
        components.label(frame, 3, 0, "最小噪声强度",
                         tooltip="指定训练期间使用的最小噪声强度。这可以帮助改进构图，但会阻止训练更精细的细节")
        components.entry(frame, 3, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(frame, 4, 0, "最大噪声强度",
                         tooltip="指定训练期间使用的最大噪声强度。这可以用来减少过拟合，但也减少了训练样本对整体图像构图的影响")
        components.entry(frame, 4, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(frame, 5, 0, "噪声权重",
                         tooltip="控制时间步长分布函数的权重参数。使用预览查看更多详细信息。")
        components.entry(frame, 5, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(frame, 6, 0, "噪声偏差",
                         tooltip="控制时间步长分布函数的偏差参数。使用预览查看更多详细信息。")
        components.entry(frame, 6, 1, self.ui_state, "noising_bias")


    def __create_masked_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # Masked Training
        components.label(frame, 0, 0, "掩码训练",
                         tooltip="掩盖训练样本，让模型专注于图像的特定部分。启用后，将为每个训练样本加载一个掩码图像。")
        components.switch(frame, 0, 1, self.ui_state, "masked_training")

        # unmasked probability
        components.label(frame, 1, 0, "未掩盖概率",
                         tooltip="启用掩码训练时，指定在未掩盖样本上进行的训练步骤数量")
        components.entry(frame, 1, 1, self.ui_state, "unmasked_probability")

        # unmasked weight
        components.label(frame, 2, 0, "未掩盖权重",
                         tooltip="启用掩码训练时，指定掩盖区域外部区域的损失权重")
        components.entry(frame, 2, 1, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(frame, 3, 0, "归一化掩盖区域损失",
                         tooltip="启用掩码训练时，根据掩盖区域的大小对每个样本的损失进行归一化")
        components.switch(frame, 3, 1, self.ui_state, "normalize_masked_area_loss")

    def __create_loss_frame(self, master, row, supports_vb_loss: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # MSE Strength
        components.label(frame, 0, 0, "MSE 强度",
                         tooltip="自定义损失设置的均方误差强度。MAE + MSE 强度通常应加起来为 1。")
        components.entry(frame, 0, 1, self.ui_state, "mse_strength")

        # MAE Strength
        components.label(frame, 1, 0, "MAE 强度",
                         tooltip="自定义损失设置的平均绝对误差强度。MAE + MSE 强度通常应加起来为 1。")
        components.entry(frame, 1, 1, self.ui_state, "mae_strength")

        # log-cosh Strength
        components.label(frame, 2, 0, "log-cosh 强度",
                         tooltip="自定义损失设置的对数 - 双曲余弦误差强度。")
        components.entry(frame, 2, 1, self.ui_state, "log_cosh_strength")

        if supports_vb_loss:
            # VB Strength
            components.label(frame, 3, 0, "VB 强度",
                             tooltip="自定义损失设置的变分下界强度。对于变分扩散模型，应设置为 1")
            components.entry(frame, 3, 1, self.ui_state, "vb_loss_strength")

        # Loss Weight function
        components.label(frame, 4, 0, "损失权重函数",
                         tooltip="损失权重函数的选择。可以帮助模型更准确地学习细节。")
        components.options(frame, 4, 1, [str(x) for x in list(LossWeight)], self.ui_state, "loss_weight_fn")

        # Loss weight strength
        components.label(frame, 5, 0, "Gamma",
                         tooltip="损失加权的逆强度。范围：1-20，仅适用于最小 SNR 和 P2。")
        components.entry(frame, 5, 1, self.ui_state, "loss_weight_strength")

        # Loss Scaler
        components.label(frame, 6, 0, "损失缩放器",
                         tooltip="选择在训练期间使用的损失缩放类型。功能上等同于：损失 * 选择")
        components.options(frame, 6, 1, [str(x) for x in list(LossScaler)], self.ui_state, "loss_scaler")

    def __open_optimizer_params_window(self):
        window = OptimizerParamsWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __open_scheduler_params_window(self):
        window = SchedulerParamsWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __open_timestep_distribution_window(self):
        window = TimestepDistributionWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __restore_optimizer_config(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

    def __restore_scheduler_config(self, variable):
        if not hasattr(self, 'lr_scheduler_adv_comp'):
            return

        if variable == "CUSTOM":
            self.lr_scheduler_adv_comp.configure(state="normal")
        else:
            self.lr_scheduler_adv_comp.configure(state="disabled")
