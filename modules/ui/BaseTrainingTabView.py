from abc import ABC, abstractmethod

from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.ui.validation_helpers import check_range, validate_resolution


class BaseTrainingTabView(ABC):
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def restore_optimizer_config(self, variable: str): pass

    @abstractmethod
    def open_optimizer_params(self): pass

    @abstractmethod
    def restore_scheduler(self, variable): pass

    @abstractmethod
    def open_scheduler_params(self): pass

    @abstractmethod
    def open_timestep_distribution(self): pass

    def build(self, column_0, column_1, column_2, controller, ui_state):
        model_type = controller.config.model_type
        if model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui(column_0, column_1, column_2, controller, ui_state)
        if model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_pixart():
            self.__setup_pixart_alpha_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_flux_1():
            self.__setup_flux_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_flux_2():
            self.__setup_flux_2_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_chroma():
            self.__setup_chroma_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_qwen():
            self.__setup_qwen_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_anima():
            self.__setup_anima_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_krea2():
            self.__setup_krea2_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_sana():
            self.__setup_sana_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_hi_dream():
            self.__setup_hi_dream_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_z_image():
            self.__setup_z_image_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_ernie():
            self.__setup_ernie_ui(column_0, column_1, column_2, controller, ui_state)
        elif model_type.is_ideogram():
            self.__setup_ideogram_ui(column_0, column_1, column_2, controller, ui_state)

    def __setup_stable_diffusion_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_layer_offloading=False)
        self.__create_embedding_frame(column_0, 2, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state, supports_circular_padding=True)
        self.__create_unet_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state, supports_generalized_offset_noise=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_stable_diffusion_3_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_n_frame(column_0, 1, ui_state, i=1, supports_include=True, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 2, ui_state, i=2, supports_include=True, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 3, ui_state, i=3, supports_include=True)
        self.__create_embedding_frame(column_0, 4, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_stable_diffusion_xl_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_n_frame(column_0, 1, ui_state, i=1, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 2, ui_state, i=2, supports_layer_offloading=False)
        self.__create_embedding_frame(column_0, 3, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state, supports_circular_padding=True)
        self.__create_unet_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state, supports_generalized_offset_noise=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_wuerstchen_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_layer_offloading=False)
        self.__create_embedding_frame(column_0, 2, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state, supports_circular_padding=True)
        self.__create_prior_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 0, ui_state)
        self.__create_loss_frame(column_2, 1, controller, ui_state)
        self.__create_layer_frame(column_2, 2, controller, ui_state)

    def __setup_pixart_alpha_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state)
        self.__create_embedding_frame(column_0, 2, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state, supports_vb_loss=True)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_flux_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_n_frame(column_0, 1, ui_state, i=1, supports_include=True, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 2, ui_state, i=2, supports_include=True, supports_sequence_length=True)
        self.__create_embedding_frame(column_0, 4, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=True)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_flux_2_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False, supports_sequence_length=True)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=True, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_chroma_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state)
        self.__create_embedding_frame(column_0, 4, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_qwen_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_anima_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_krea2_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_z_image_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_ernie_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_ideogram_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state, supports_clip_skip=False, supports_training=False, supports_dropout=False)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_unconditional_transformer_frame(column_1, 2, ui_state)
        self.__create_noise_frame(column_1, 3, ui_state, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_sana_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_frame(column_0, 1, ui_state)
        self.__create_embedding_frame(column_0, 2, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state)
        self.__create_transformer_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_hunyuan_video_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_n_frame(column_0, 1, ui_state, i=1, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 2, ui_state, i=2, supports_include=True, supports_layer_offloading=False)
        self.__create_embedding_frame(column_0, 4, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state, video_training_enabled=True)
        self.__create_transformer_frame(column_1, 1, ui_state, supports_guidance_scale=True)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __setup_hi_dream_ui(self, column_0, column_1, column_2, controller, ui_state):
        self.__create_base_frame(column_0, 0, controller, ui_state)
        self.__create_text_encoder_n_frame(column_0, 1, ui_state, i=1, supports_include=True, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 2, ui_state, i=2, supports_include=True, supports_layer_offloading=False)
        self.__create_text_encoder_n_frame(column_0, 3, ui_state, i=3, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 4, ui_state, i=4, supports_include=True, supports_layer_skip=False)
        self.__create_embedding_frame(column_0, 5, ui_state)

        self.__create_base2_frame(column_1, 0, controller, ui_state, video_training_enabled=True)
        self.__create_transformer_frame(column_1, 1, ui_state)
        self.__create_noise_frame(column_1, 2, ui_state)

        self.__create_masked_frame(column_2, 1, ui_state)
        self.__create_loss_frame(column_2, 2, controller, ui_state)
        self.__create_layer_frame(column_2, 3, controller, ui_state)

    def __create_base_frame(self, master, row, controller, ui_state):
        frame = self.components.section_frame(master, row)

        # optimizer
        self.components.label(frame, 0, 0, "优化器",
                              tooltip="优化器类型")
        self.components.options_adv(frame, 0, 1, [str(x) for x in list(Optimizer)], ui_state, "optimizer.optimizer",
                                    command=self.restore_optimizer_config,
                                    adv_command=self.open_optimizer_params)

        # learning rate scheduler
        # Wackiness will ensue when reloading configs if we don't check and clear this first.
        if hasattr(self, "lr_scheduler_comp"):
            delattr(self, "lr_scheduler_comp")
            delattr(self, "lr_scheduler_adv_comp")
        self.components.label(frame, 1, 0, "学习率调度器",
                              tooltip="训练过程中自动调整学习率的调度器")
        _, d = self.components.options_adv(frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], ui_state,
                                           "learning_rate_scheduler",
                                           command=self.restore_scheduler,
                                           adv_command=self.open_scheduler_params)
        self.lr_scheduler_comp = d['component']
        self.lr_scheduler_adv_comp = d['button_component']
        # Initial call requires the presence of self.lr_scheduler_adv_comp.
        self.restore_scheduler(ui_state.get_var("learning_rate_scheduler").get())

        # learning rate
        self.components.label(frame, 2, 0, "学习率",
                              tooltip="基础学习率")
        self.components.entry(frame, 2, 1, ui_state, "learning_rate", required=True)

        # learning rate warmup steps
        self.components.label(frame, 3, 0, "学习率预热步数",
                              tooltip="学习率从0渐增到指定值的步数，>1为固定步数，<=1为总步数百分比")
        self.components.entry(frame, 3, 1, ui_state, "learning_rate_warmup_steps")

        # learning rate min factor
        self.components.label(frame, 4, 0, "学习率最小因子",
                              tooltip="浮点数，百分比方式。如0.1则最终学习率为初始值的10%")
        self.components.entry(frame, 4, 1, ui_state, "learning_rate_min_factor",
                              extra_validate=check_range(lower=0, upper=0.99, message="学习率最小因子必须在0到0.99之间"))

        # learning rate cycles
        self.components.label(frame, 5, 0, "学习率周期数",
                              tooltip="学习率周期数，仅调度器支持时有效")
        self.components.entry(frame, 5, 1, ui_state, "learning_rate_cycles")

        # epochs
        self.components.label(frame, 6, 0, "训练轮数",
                              tooltip="完整训练运行的轮数")
        self.components.entry(frame, 6, 1, ui_state, "epochs", required=True)

        # batch size
        self.components.label(frame, 7, 0, "本地批次大小",
                              tooltip="单步训练的批次大小。多GPU时每块GPU的批次大小")
        self.components.entry(frame, 7, 1, ui_state, "batch_size", required=True)

        # accumulation steps
        self.components.label(frame, 8, 0, "梯度累积步数",
                              tooltip="梯度累积步数，增加此值以训练速度换取更大批次")
        self.components.entry(frame, 8, 1, ui_state, "gradient_accumulation_steps", required=True)

        # Learning Rate Scaler
        self.components.label(frame, 9, 0, "学习率缩放器",
                              tooltip="学习率缩放类型，等效于: LR * SQRT(选择值)")
        self.components.options(frame, 9, 1, [str(x) for x in list(LearningRateScaler)], ui_state,
                                "learning_rate_scaler")

        # clip grad norm
        self.components.label(frame, 10, 0, "梯度裁剪",
                              tooltip="梯度范数裁剪，留空则禁用")
        self.components.entry(frame, 10, 1, ui_state, "clip_grad_norm")

    def __create_base2_frame(self, master, row, controller, ui_state, video_training_enabled: bool = False,
                              supports_circular_padding: bool = False):
        frame = self.components.section_frame(master, row)
        row = 0

        # attention mechanism
        self.components.label(frame, row, 0, "注意力机制",
                              tooltip="训练使用的注意力机制。Linux用torch SDPA，Windows可手动安装flash-attn")
        self.components.options_kv(frame, row, 1, controller.get_attention_mechanisms(), ui_state,
                                   "attention_mechanism")
        row += 1

        # ema
        self.components.label(frame, row, 0, "EMA",
                              tooltip="EMA对多步训练取平均，更好地保留大数据集中的不同概念")
        self.components.options(frame, row, 1, [str(x) for x in list(EMAMode)], ui_state, "ema")
        row += 1

        # ema decay
        self.components.label(frame, row, 0, "EMA衰减",
                              tooltip="EMA模型衰减参数。大数据集设0.9999，小数据集设0.999或0.998")
        self.components.entry(frame, row, 1, ui_state, "ema_decay",
                              extra_validate=check_range(lower=0.5, upper=1,
                                                        message="EMA衰减必须在0.5到1之间"))
        row += 1

        # ema update step interval
        self.components.label(frame, row, 0, "EMA更新步间隔",
                              tooltip="EMA更新之间的步数")
        self.components.entry(frame, row, 1, ui_state, "ema_update_step_interval")
        row += 1

        # train dtype
        self.components.label(frame, row, 0, "训练数据类型",
                              tooltip="训练混合精度数据类型，可提高速度但降低精度")
        self.components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], ui_state, "train_dtype")
        row += 1

        # fallback train dtype
        self.components.label(frame, row, 0, "回退训练数据类型",
                              tooltip="不支持float16的训练阶段的混合精度数据类型")
        self.components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "fallback_train_dtype")
        row += 1

        # autocast cache
        self.components.label(frame, row, 0, "自动转换缓存",
                              tooltip="启用自动转换缓存，禁用可减少内存但增加训练时间")
        self.components.switch(frame, row, 1, ui_state, "enable_autocast_cache")
        row += 1

        # resolution
        self.components.label(frame, row, 0, "分辨率",
                              tooltip="训练分辨率，可用逗号分隔多个分辨率，或指定 <宽>x<高> 格式")
        self.components.entry(frame, row, 1, ui_state, "resolution", required=True,
                              extra_validate=validate_resolution())
        row += 1

        # frames
        if video_training_enabled:
            self.components.label(frame, row, 0, "帧数",
                                  tooltip="训练使用的帧数")
            self.components.entry(frame, row, 1, ui_state, "frames", required=True)
            row += 1

        # force circular padding
        if supports_circular_padding:
            self.components.label(frame, row, 0, "强制循环填充",
                                  tooltip="为所有卷积层启用循环填充，更好地训练无缝图像")
            self.components.switch(frame, row, 1, ui_state, "force_circular_padding")

    def __create_offloading_widgets(self, frame, row, ui_state, part, supports_checkpointing=True,
                                    supports_activation_offloading=False, supports_layer_offloading=True):
        if supports_checkpointing:
            self.components.label(frame, row, 0, "梯度检查点",
                                  tooltip="启用梯度检查点，减少显存占用但降低训练速度")
            self.components.switch(frame, row, 1, ui_state, f"{part}.gradient_checkpointing")
            row += 1

        if supports_layer_offloading:
            self.components.label(frame, row, 0, "层卸载比例",
                                  tooltip="卸载到CPU的层比例，0=禁用，1=全部")
            self.components.entry(frame, row, 1, ui_state, f"{part}.offload_fraction")
            row += 1

        if supports_activation_offloading:
            self.components.label(frame, row, 0, "卸载激活值",
                                  tooltip="训练时将激活值卸载到CPU以减少显存占用")
            self.components.switch(frame, row, 1, ui_state, f"{part}.activation_offloading")
            row += 1

        return row

    def __create_text_encoder_frame(self, master, row, ui_state, supports_clip_skip=True, supports_training=True,
                                    supports_sequence_length=False, supports_dropout=True, supports_layer_offloading=True):
        frame = self.components.section_frame(master, row)
        row = 0

        if supports_training:
            self.components.label(frame, row, 0, "训练文本编码器",
                                  tooltip="启用文本编码器训练")
            self.components.switch(frame, row, 1, ui_state, "text_encoder.train")
            row += 1
        else:
            # no Train switch to act as the frame's header, so add an explicit one
            self.components.label(frame, row, 0, "文本编码器")
            row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, "text_encoder", supports_checkpointing=supports_training,
                                               supports_layer_offloading=supports_layer_offloading)

        if supports_dropout:
            # dropout
            self.components.label(frame, row, 0, "标签丢弃概率",
                                  tooltip="丢弃文本编码器条件的概率")
            self.components.entry(frame, row, 1, ui_state, "text_encoder.dropout_probability")
            row += 1

        if supports_training:
            # train text encoder epochs
            self.components.label(frame, row, 0, "训练停止条件",
                                  tooltip="何时停止训练文本编码器")
            self.components.time_entry(frame, row, 1, ui_state, "text_encoder.stop_training_after",
                                       "text_encoder.stop_training_after_unit", supports_time_units=False)
            row += 1

            # text encoder learning rate
            self.components.label(frame, row, 0, "文本编码器学习率",
                                  tooltip="文本编码器学习率，覆盖基础学习率")
            self.components.entry(frame, row, 1, ui_state, "text_encoder.learning_rate")
            row += 1

        if supports_clip_skip:
            # text encoder layer skip (clip skip)
            self.components.label(frame, row, 0, "Clip跳层",
                                  tooltip="额外跳过的Clip层数，0为模型默认")
            self.components.entry(frame, row, 1, ui_state, "text_encoder_layer_skip")
            row += 1

        if supports_sequence_length:
            # text encoder sequence length
            self.components.label(frame, row, 0, "文本编码器序列长度",
                                  tooltip="标签Token数")
            self.components.entry(frame, row, 1, ui_state, "text_encoder_sequence_length")
            row += 1

    def __create_text_encoder_n_frame(
            self,
            master,
            row: int,
            ui_state,
            i: int,
            supports_include: bool = False,
            supports_layer_skip: bool = True,
            supports_sequence_length: bool = False,
            supports_layer_offloading: bool = True,
    ):
        frame = self.components.section_frame(master, row)
        row = 0

        suffix = f"_{i}" if i > 1 else ""

        if supports_include:
            # include text encoder
            self.components.label(frame, row, 0, f"包含文本编码器{i}",
                                  tooltip=f"在训练中包含文本编码器{i}")
            self.components.switch(frame, row, 1, ui_state, f"text_encoder{suffix}.include")
            row += 1

        # train text encoder
        self.components.label(frame, row, 0, f"训练文本编码器{i}",
                              tooltip=f"启用文本编码器{i}训练")
        self.components.switch(frame, row, 1, ui_state, f"text_encoder{suffix}.train")
        row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, f"text_encoder{suffix}",
                                               supports_layer_offloading=supports_layer_offloading)

        # train text encoder embedding
        self.components.label(frame, row, 0, f"训练文本编码器{i}嵌入",
                              tooltip=f"启用文本编码器{i}嵌入训练")
        self.components.switch(frame, row, 1, ui_state, f"text_encoder{suffix}.train_embedding")
        row += 1

        # dropout
        self.components.label(frame, row, 0, "丢弃概率",
                              tooltip=f"丢弃文本编码器{i}条件的概率")
        self.components.entry(frame, row, 1, ui_state, f"text_encoder{suffix}.dropout_probability")
        row += 1

        # train text encoder epochs
        self.components.label(frame, row, 0, "训练停止条件",
                              tooltip=f"何时停止训练文本编码器{i}")
        self.components.time_entry(frame, row, 1, ui_state, f"text_encoder{suffix}.stop_training_after",
                                   f"text_encoder{suffix}.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        self.components.label(frame, row, 0, f"文本编码器{i}学习率",
                              tooltip=f"文本编码器{i}学习率，覆盖基础学习率")
        self.components.entry(frame, row, 1, ui_state, f"text_encoder{suffix}.learning_rate")
        row += 1

        if supports_layer_skip:
            # text encoder layer skip (clip skip)
            self.components.label(frame, row, 0, f"文本编码器{i} Clip跳层",
                                  tooltip="额外跳过的Clip层数，0为模型默认")
            self.components.entry(frame, row, 1, ui_state, f"text_encoder{suffix}_layer_skip")
            row += 1

        if supports_sequence_length:
            # text encoder sequence length
            self.components.label(frame, row, 0, f"文本编码器{i}序列长度",
                                  tooltip="覆盖标签Token数，留空使用模型默认值")
            self.components.entry(frame, row, 1, ui_state, f"text_encoder{suffix}_sequence_length")
            row += 1

    def __create_embedding_frame(self, master, row, ui_state):
        frame = self.components.section_frame(master, row)

        # embedding learning rate
        self.components.label(frame, 0, 0, "嵌入学习率",
                              tooltip="嵌入学习率，覆盖基础学习率")
        self.components.entry(frame, 0, 1, ui_state, "embedding_learning_rate")

        # preserve embedding norm
        self.components.label(frame, 1, 0, "保留嵌入范数",
                              tooltip="将每个训练嵌入重缩放至中位嵌入范数")
        self.components.switch(frame, 1, 1, ui_state, "preserve_embedding_norm")

    def __create_unet_frame(self, master, row, ui_state):
        frame = self.components.section_frame(master, row)
        row = 0

        # train unet
        self.components.label(frame, row, 0, "训练UNet",
                              tooltip="启用UNet模型训练")
        self.components.switch(frame, row, 1, ui_state, "unet.train")
        row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, "unet", supports_layer_offloading=False)

        # train unet epochs
        self.components.label(frame, row, 0, "训练停止条件",
                              tooltip="何时停止训练UNet")
        self.components.time_entry(frame, row, 1, ui_state, "unet.stop_training_after", "unet.stop_training_after_unit",
                                   supports_time_units=False)
        row += 1

        # unet learning rate
        self.components.label(frame, row, 0, "UNet学习率",
                              tooltip="UNet学习率，覆盖基础学习率")
        self.components.entry(frame, row, 1, ui_state, "unet.learning_rate")
        row += 1

        # rescale noise scheduler to zero terminal SNR
        self.components.label(frame, row, 0, "重缩放噪声调度+V预测",
                              tooltip="将噪声调度器重缩放至零终端信噪比，切换模型到v预测目标",
                              wraplength=130)
        self.components.switch(frame, row, 1, ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")
        row += 1

    def __create_prior_frame(self, master, row, ui_state):
        frame = self.components.section_frame(master, row)
        row = 0

        # train prior
        self.components.label(frame, row, 0, "训练Prior",
                              tooltip="启用Prior模型训练")
        self.components.switch(frame, row, 1, ui_state, "prior.train")
        row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, "prior", supports_layer_offloading=False)

        # train prior epochs
        self.components.label(frame, row, 0, "训练停止条件",
                              tooltip="何时停止训练Prior")
        self.components.time_entry(frame, row, 1, ui_state, "prior.stop_training_after",
                                   "prior.stop_training_after_unit", supports_time_units=False)
        row += 1

        # prior learning rate
        self.components.label(frame, row, 0, "Prior学习率",
                              tooltip="Prior学习率，覆盖基础学习率")
        self.components.entry(frame, row, 1, ui_state, "prior.learning_rate")
        row += 1

    def __create_transformer_frame(self, master, row, ui_state, supports_guidance_scale: bool = False,
                                   supports_force_attention_mask: bool = True):
        frame = self.components.section_frame(master, row)
        row = 0

        # train transformer
        self.components.label(frame, row, 0, "训练Transformer",
                              tooltip="启用Transformer模型训练")
        self.components.switch(frame, row, 1, ui_state, "transformer.train")
        row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, "transformer", supports_activation_offloading=True)

        # train transformer epochs
        self.components.label(frame, row, 0, "训练停止条件",
                              tooltip="何时停止训练Transformer")
        self.components.time_entry(frame, row, 1, ui_state, "transformer.stop_training_after",
                                   "transformer.stop_training_after_unit", supports_time_units=False)
        row += 1

        # transformer learning rate
        self.components.label(frame, row, 0, "Transformer学习率",
                              tooltip="Transformer学习率，覆盖基础学习率")
        self.components.entry(frame, row, 1, ui_state, "transformer.learning_rate")
        row += 1

        if supports_force_attention_mask:
            # transformer learning rate
            self.components.label(frame, row, 0, "强制注意力遮罩",
                                  tooltip="强制向Transformer传递文本嵌入注意力遮罩，可改善短标签训练")
            self.components.switch(frame, row, 1, ui_state, "transformer.attention_mask")
            row += 1

        if supports_guidance_scale:
            # guidance scale
            self.components.label(frame, row, 0, "引导尺度",
                                  tooltip="引导蒸馏模型传递给Transformer的引导尺度")
            self.components.entry(frame, row, 1, ui_state, "transformer.guidance_scale")
            row += 1

    def __create_unconditional_transformer_frame(self, master, row, ui_state):
        frame = self.components.section_frame(master, row)
        row = 0

        # include unconditional transformer
        self.components.label(frame, row, 0, "包含无条件Transformer",
                              tooltip="Loads the dedicated unconditional transformer used for the negative branch of CFG "
                                      "during sampling. If disabled, CFG above 1.0 still works by running an empty prompt "
                                      "through the conditional transformer instead, at reduced VRAM and load time")
        self.components.switch(frame, row, 1, ui_state, "unconditional_transformer.include")
        row += 1

        row = self.__create_offloading_widgets(frame, row, ui_state, "unconditional_transformer", supports_checkpointing=False)

    def __create_noise_frame(self, master, row, ui_state,
                              supports_generalized_offset_noise: bool = False,
                              supports_dynamic_timestep_shifting: bool = False):
        frame = self.components.section_frame(master, row)

        # offset noise weight
        self.components.label(frame, 0, 0, "偏移噪声权重",
                              tooltip="每步训练添加的偏移噪声权重")
        self.components.entry(frame, 0, 1, ui_state, "offset_noise_weight")

        if supports_generalized_offset_noise:
            # generalized offset noise weight
            self.components.label(frame, 1, 0, "广义偏移噪声",
                                  tooltip="逐时间步的亮度调节，训练更稳定。建议从0.02开始",
                                  wraplength=130)
            self.components.switch(frame, 1, 1, ui_state, "generalized_offset_noise")

        # perturbation noise weight
        self.components.label(frame, 2, 0, "扰动噪声权重",
                              tooltip="每步训练添加的扰动噪声权重")
        self.components.entry(frame, 2, 1, ui_state, "perturbation_noise_weight")

        # timestep distribution
        self.components.label(frame, 3, 0, "时间步分布",
                              tooltip="选择训练时的时间步采样函数",
                              wide_tooltip=True)
        self.components.options_adv(frame, 3, 1, [str(x) for x in list(TimestepDistribution)], ui_state,
                                    "timestep_distribution",
                                    adv_command=self.open_timestep_distribution)

        # min noising strength
        self.components.label(frame, 4, 0, "最小噪声强度",
                              tooltip="训练最小噪声强度，有助于构图但会阻碍细节训练")
        self.components.entry(frame, 4, 1, ui_state, "min_noising_strength", required=True)

        # max noising strength
        self.components.label(frame, 5, 0, "最大噪声强度",
                              tooltip="训练最大噪声强度，可减少过拟合但降低样本对构图的影响")
        self.components.entry(frame, 5, 1, ui_state, "max_noising_strength", required=True)

        # noising weight
        self.components.label(frame, 6, 0, "噪声权重",
                              tooltip="控制时间步分布函数的权重参数")
        self.components.entry(frame, 6, 1, ui_state, "noising_weight", required=True)

        # noising bias
        self.components.label(frame, 7, 0, "噪声偏差",
                              tooltip="控制时间步分布函数的偏差参数")
        self.components.entry(frame, 7, 1, ui_state, "noising_bias", required=True)

        # timestep shift
        self.components.label(frame, 8, 0, "时间步偏移",
                              tooltip="偏移时间步分布，使用预览查看详情")
        self.components.entry(frame, 8, 1, ui_state, "timestep_shift", required=True)

        if supports_dynamic_timestep_shifting:
            # dynamic timestep shifting
            self.components.label(frame, 9, 0, "动态时间步偏移",
                                  tooltip="Dynamically shift the timestep distribution based on resolution. If enabled, the shifting parameters are taken from the model's scheduler configuration and Timestep Shift is ignored. For Ideogram, the shifting instead follows the model's own resolution-aware sampling schedule. Note: For Z-Image, the dynamic shifting parameters are likely wrong and unknown. Use with care or set your own, fixed shift.", wide_tooltip=True)
            self.components.switch(frame, 9, 1, ui_state, "dynamic_timestep_shifting")

    def __create_masked_frame(self, master, row, ui_state):
        frame = self.components.section_frame(master, row)

        # Masked Training
        self.components.label(frame, 0, 0, "遮罩训练",
                              tooltip="Masks the training samples to let the model focus on certain parts of the image. When enabled, one mask image is loaded for each training sample.")
        self.components.switch(frame, 0, 1, ui_state, "masked_training")

        # unmasked probability
        self.components.label(frame, 1, 0, "未遮罩概率",
                              tooltip="遮罩训练时未遮罩样本的训练步数")
        self.components.entry(frame, 1, 1, ui_state, "unmasked_probability",
                              extra_validate=check_range(lower=0, upper=1, message="未遮罩概率必须在0到1之间"))

        # unmasked weight
        self.components.label(frame, 2, 0, "未遮罩权重",
                              tooltip="遮罩训练时遮罩外区域的损失权重")
        self.components.entry(frame, 2, 1, ui_state, "unmasked_weight",
                              extra_validate=check_range(lower=0, upper=1, message="未遮罩权重必须在0到1之间"))

        # normalize masked area loss
        self.components.label(frame, 3, 0, "归一化遮罩区域损失",
                              tooltip="遮罩训练时按遮罩区域大小归一化损失")
        self.components.switch(frame, 3, 1, ui_state, "normalize_masked_area_loss")

        # masked prior preservation
        self.components.label(frame, 4, 0, "遮罩Prior保留权重",
                              tooltip="使用原始未训练模型输出保留遮罩外区域，仅限LoRA训练")
        self.components.entry(frame, 4, 1, ui_state, "masked_prior_preservation_weight",
                              extra_validate=check_range(lower=0, upper=1, message="遮罩Prior保留权重必须在0到1之间"))

        # use custom conditioning image
        self.components.label(frame, 5, 0, "自定义条件图像",
                              tooltip="启用自定义条件图像，适用于对象移除等特殊场景")
        self.components.switch(frame, 5, 1, ui_state, "custom_conditioning_image")

    def __create_loss_frame(self, master, row, controller, ui_state,
                            supports_vb_loss: bool = False):
        frame = self.components.section_frame(master, row)

        # MSE Strength
        self.components.label(frame, 0, 0, "MSE强度",
                              tooltip="均方误差强度，强度总和应为1")
        self.components.entry(frame, 0, 1, ui_state, "mse_strength", required=True)

        # MAE Strength
        self.components.label(frame, 1, 0, "MAE强度",
                              tooltip="平均绝对误差强度，强度总和应为1")
        self.components.entry(frame, 1, 1, ui_state, "mae_strength", required=True)

        # log-cosh Strength
        self.components.label(frame, 2, 0, "log-cosh Strength",
                              tooltip="Log - Hyperbolic cosine Error strength for custom loss settings. Strengths should generally sum to 1.")
        self.components.entry(frame, 2, 1, ui_state, "log_cosh_strength", required=True)

        # Huber Strength
        self.components.label(frame, 3, 0, "Huber强度",
                              tooltip="Huber损失强度，比MSE对异常值更不敏感")
        self.components.entry(frame, 3, 1, ui_state, "huber_strength", required=True)

        # Huber Delta
        self.components.label(frame, 4, 0, "Huber Delta",
                              tooltip="Huber损失的delta参数")
        self.components.entry(frame, 4, 1, ui_state, "huber_delta", required=True)

        if supports_vb_loss:
            # VB Strength
            self.components.label(frame, 5, 0, "VB强度",
                                  tooltip="变分下界强度，变分扩散模型应设为1")
            self.components.entry(frame, 5, 1, ui_state, "vb_loss_strength", required=True)

        # Loss Weight function
        self.components.label(frame, 6, 0, "损失权重函数",
                              tooltip="损失权重函数选择，帮助模型更准确学习细节")
        self.components.options(frame, 6, 1, [str(x) for x in list(LossWeight)
                                              if x.supports_flow_matching() == controller.is_flow_matching()
                                              or x == LossWeight.CONSTANT
                                              ],
                                ui_state, "loss_weight_fn")

        row = 7

        # Loss weight strength
        if not controller.is_flow_matching():
            self.components.label(frame, row, 0, "Gamma",
                                  tooltip="损失权重逆强度，范围1-20，仅用于Min SNR和P2")
            self.components.entry(frame, row, 1, ui_state, "loss_weight_strength",
                                  extra_validate=check_range(lower=1, upper=20, message="Gamma必须在1到20之间"))
            row += 1

        # Loss Scaler
        self.components.label(frame, row, 0, "损失缩放器",
                              tooltip="训练损失缩放类型，等效于: Loss * 选择值")
        self.components.options(frame, row, 1, [str(x) for x in list(LossScaler)], ui_state, "loss_scaler")
        row += 1

    def __create_layer_frame(self, master, row, controller, ui_state):
        presets = controller.get_layer_presets()
        self.components.layer_filter_entry(master, row, 0, ui_state,
                                           preset_var_name="layer_filter_preset", presets=presets,
                                           preset_label="层过滤器",
                                           preset_tooltip="Select a preset defining which layers to train, or select 'Custom' to define your own.\nA blank 'custom' field or 'Full' will train all layers.",
                                           entry_var_name="layer_filter",
                                           entry_tooltip="逗号分隔的训练层列表，支持正则表达式",
                                           regex_var_name="layer_filter_regex",
                                           regex_tooltip="启用后层过滤器使用正则匹配，否则使用子串匹配",
                                           )
