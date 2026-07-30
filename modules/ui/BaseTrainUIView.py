from abc import ABC, abstractmethod
from collections.abc import Callable

from modules.util import path_util
from modules.util.enum.DataType import DataType
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.PathIOType import PathIOType


class BaseTrainUIView(ABC):
    def __init__(self, components, controller, ui_state):
        self.components = components
        self.controller = controller
        self.ui_state = ui_state

    # --- Abstract callbacks (controller calls into view) ---

    @abstractmethod
    def on_update_status(self, status: str): pass

    @abstractmethod
    def on_training_started(self): pass

    @abstractmethod
    def on_training_stopped(self, error_caught: bool): pass

    @abstractmethod
    def on_training_stopping(self): pass

    @abstractmethod
    def on_update_progress(self, epoch_step: int, max_step: int, epoch: int, max_epoch: int, eta_str: str | None): pass

    @abstractmethod
    def schedule_on_main_thread(self, fn: Callable): pass

    @abstractmethod
    def get_cloud_reattach(self) -> bool: pass

    @abstractmethod
    def save_default(self): pass

    @abstractmethod
    def show_validation_errors(self, errors: list[str]): pass

    @abstractmethod
    def wait_window(self, window): pass

    @abstractmethod
    def show_window(self, window): pass

    @abstractmethod
    def connect_window_closed(self, window, callback): pass

    def sync_cloud_secrets(self):
        # Called from training thread — defer to main thread
        self.schedule_on_main_thread(
            lambda: self.ui_state.get_var("secrets.cloud").update(self.controller.train_config.secrets.cloud)
        )

    def start_training(self):
        self.controller.start_training()

    def open_tensorboard(self):
        self.controller.open_tensorboard()

    def sample_now(self):
        self.controller.sample_now()

    def backup_now(self):
        self.controller.backup_now()

    def save_now(self):
        self.controller.save_now()

    @abstractmethod
    def open_dataset_tool(self): pass

    @abstractmethod
    def open_video_tool(self): pass

    @abstractmethod
    def open_convert_model_tool(self): pass

    @abstractmethod
    def open_sampling_tool(self): pass

    @abstractmethod
    def open_manual_sample_window(self): pass

    @abstractmethod
    def open_profiling_tool(self): pass

    @abstractmethod
    def export_training(self): pass

    @abstractmethod
    def generate_debug_package(self): pass

    # --- Content builders (components calls; called by CTK view after frame creation) ---

    def build_bottom_bar_content(self, frame, status_frame, controller, ui_state):
        self.set_step_progress, self.set_epoch_progress = self.components.double_progress(frame, 0, 0, "step", "epoch")

        self.status_label = self.components.label(status_frame, 0, 0, "", pad=0,
                                             tooltip="训练运行当前状态")
        self.eta_label = self.components.label(status_frame, 1, 0, "", pad=0)

        self.export_button = self.components.button(frame, 0, 3, "导出", self.export_training,
                                             width=60, padx=5, pady=(15, 0),
                                             tooltip="导出当前配置为无UI运行脚本")

        self.components.button(frame, 0, 4, "调试", self.generate_debug_package,
                                           width=60, padx=(5, 25), pady=(15, 0),
                                           tooltip="生成包含配置和调试报告的zip文件，用于报告问题")

        self.components.button(frame, 0, 5, "Tensorboard", self.open_tensorboard,
                                           width=100, padx=(0, 5), pady=(15, 0))

        self.training_button = self.components.button(frame, 0, 6, "开始训练", self.start_training,
                                                 padx=(5, 20), pady=(15, 0))

    def build_general_tab_content(self, frame, controller, ui_state):
        # workspace dir
        self.components.label(frame, 0, 0, "工作空间目录",
                         tooltip="此训练运行所有文件保存的目录")
        self.components.path_entry(frame, 0, 1, ui_state, "workspace_dir", mode="dir", command=controller._on_workspace_dir_change)

        # cache dir
        self.components.label(frame, 0, 2, "缓存目录",
                         tooltip="缓存数据保存的目录")
        self.components.path_entry(frame, 0, 3, ui_state, "cache_dir", mode="dir")

        # continue from previous backup
        self.components.label(frame, 2, 0, "从上次备份继续",
                         tooltip="自动从<workspace>/backup中的上次备份继续训练")
        self.components.switch(frame, 2, 1, ui_state, "continue_last_backup")

        # only cache
        self.components.label(frame, 2, 2, "仅缓存",
                         tooltip="仅填充缓存，不进行训练")
        self.components.switch(frame, 2, 3, ui_state, "only_cache")

        # TODO: In Phase 4 rework the general tab.
        # prevent overwrites
        self.components.label(frame, 3, 0, "防止覆盖",
                         tooltip="启用后，已存在的输出路径将被标记为无效以防止意外覆盖")
        self.components.switch(frame, 3, 1, ui_state, "prevent_overwrites")

        # debug
        self.components.label(frame, 4, 0, "调试模式",
                         tooltip="训练时将调试信息保存到调试目录")
        self.components.switch(frame, 4, 1, ui_state, "debug_mode")

        self.components.label(frame, 4, 2, "调试目录",
                         tooltip="调试数据保存的目录")
        self.components.path_entry(frame, 4, 3, ui_state, "debug_dir", mode="dir", io_type=PathIOType.OUTPUT)

        # tensorboard
        self.components.label(frame, 6, 0, "Tensorboard",
                         tooltip="训练时启动Tensorboard Web UI")
        self.components.switch(frame, 6, 1, ui_state, "tensorboard")

        self.components.label(frame, 6, 2, "常驻Tensorboard",
                         tooltip="非训练时也保持Tensorboard可访问")
        self.components.switch(frame, 6, 3, ui_state, "tensorboard_always_on", command=controller._on_always_on_tensorboard_toggle)

        self.components.label(frame, 7, 0, "暴露Tensorboard",
                         tooltip="将Tensorboard暴露到所有网络接口")
        self.components.switch(frame, 7, 1, ui_state, "tensorboard_expose")
        self.components.label(frame, 7, 2, "Tensorboard端口",
                         tooltip="Tensorboard链接端口")
        self.components.entry(frame, 7, 3, ui_state, "tensorboard_port")

        # validation
        self.components.label(frame, 8, 0, "验证",
                         tooltip="启用验证步骤并在Tensorboard添加图表")
        self.components.switch(frame, 8, 1, ui_state, "validation")

        self.components.label(frame, 8, 2, "验证间隔",
                         tooltip="训练验证间隔")
        self.components.time_entry(frame, 8, 3, ui_state, "validate_after", "validate_after_unit")

        # device
        self.components.label(frame, 10, 0, "数据加载线程",
                         tooltip="数据加载线程数，缓存时GPU有余量可增加")
        self.components.entry(frame, 10, 1, ui_state, "dataloader_threads", required=True)

        self.components.label(frame, 11, 0, "训练设备",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\". Must be \"cuda\" for multi-GPU training.")
        self.components.entry(frame, 11, 1, ui_state, "train_device", required=True)

        self.components.label(frame, 11, 2, "异步卸载",
                         tooltip="使用CUDA流重叠CPU<->GPU传输与计算")
        self.components.switch(frame, 11, 3, ui_state, "async_offloading")

        self.components.label(frame, 12, 0, "Multi-GPU",
                         tooltip="启用多GPU训练")
        self.components.switch(frame, 12, 1, ui_state, "multi_gpu")
        self.components.label(frame, 12, 2, "设备索引",
                         tooltip="多GPU：逗号分隔的设备索引列表，留空使用所有GPU")
        self.components.entry(frame, 12, 3, ui_state, "device_indexes")

        self.components.label(frame, 13, 0, "梯度归约精度",
                         tooltip="WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
                                 "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
                                 "FLOAT_32: Reduce gradients in float32\n"
                                 "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
                         wide_tooltip=True)
        self.components.options(frame, 13, 1, [str(x) for x in list(GradientReducePrecision)], ui_state,
                           "gradient_reduce_precision")

        self.components.label(frame, 13, 2, "融合梯度归约",
                         tooltip="多GPU：反向传播时的梯度同步，配合异步梯度归约更高效")
        self.components.switch(frame, 13, 3, ui_state, "fused_gradient_reduce")

        self.components.label(frame, 14, 0, "异步梯度归约",
                         tooltip="多GPU：反向传播时异步启动梯度归约，更高效但占用显存")
        self.components.switch(frame, 14, 1, ui_state, "async_gradient_reduce")
        self.components.label(frame, 14, 2, "Buffer size (MB)",
                         tooltip="Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"")
        self.components.entry(frame, 14, 3, ui_state, "async_gradient_reduce_buffer")

        self.components.label(frame, 15, 0, "临时设备",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")
        self.components.entry(frame, 15, 1, ui_state, "temp_device")

    def build_data_tab_content(self, frame, controller, ui_state):
        # aspect ratio bucketing
        self.components.label(frame, 0, 0, "宽高比分桶",
                         tooltip="宽高比分桶允许在不同宽高比的图像上训练")
        self.components.switch(frame, 0, 1, ui_state, "aspect_ratio_bucketing")

        # latent caching
        self.components.label(frame, 1, 0, "潜在缓存",
                         tooltip="缓存可在轮次间复用的中间训练数据")
        self.components.switch(frame, 1, 1, ui_state, "latent_caching")

        # clear cache before training
        self.components.label(frame, 2, 0, "训练前清除缓存",
                         tooltip="训练前清除缓存目录，仅在使用相同缓存数据时禁用")
        self.components.switch(frame, 2, 1, ui_state, "clear_cache_before_training")

    def build_sampling_tab_header(self, top_frame, sub_frame, controller, ui_state):
        self.components.label(top_frame, 0, 0, "采样间隔",
                         tooltip="训练时自动采样的间隔")
        self.components.time_entry(top_frame, 0, 1, ui_state, "sample_after", "sample_after_unit")

        self.components.label(top_frame, 0, 2, "跳过首个",
                         tooltip="经过此间隔后自动开始采样")
        self.components.entry(top_frame, 0, 3, ui_state, "sample_skip_first", width=50, sticky="nw")

        self.components.label(top_frame, 0, 4, "Format",
                         tooltip="保存样本的文件格式")
        self.components.options_kv(top_frame, 0, 5, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], ui_state, "sample_image_format")

        self.components.button(top_frame, 0, 6, "sample now", self.sample_now)

        self.components.button(top_frame, 0, 7, "manual sample", self.open_manual_sample_window)

        self.components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                         tooltip="使用EMA时是否包含非EMA采样")
        self.components.switch(sub_frame, 0, 1, ui_state, "non_ema_sampling")

        self.components.label(sub_frame, 0, 2, "采样到Tensorboard",
                         tooltip="是否在Tensorboard输出中包含采样图像")
        self.components.switch(sub_frame, 0, 3, ui_state, "samples_to_tensorboard")

    def build_backup_tab_content(self, frame, controller, ui_state):
        # backup after
        self.components.label(frame, 0, 0, "备份间隔",
                         tooltip="训练时自动创建模型备份的间隔")
        self.components.time_entry(frame, 0, 1, ui_state, "backup_after", "backup_after_unit")

        # backup now
        self.components.button(frame, 0, 3, "backup now", self.backup_now)

        # rolling backup
        self.components.label(frame, 1, 0, "滚动备份",
                         tooltip="启用滚动备份后自动删除旧备份")
        self.components.switch(frame, 1, 1, ui_state, "rolling_backup")

        # rolling backup count
        self.components.label(frame, 2, 0, "滚动备份数量",
                         tooltip="滚动备份保留的数量")
        self.components.entry(frame, 2, 1, ui_state, "rolling_backup_count")

        # backup before save
        self.components.label(frame, 3, 0, "保存前备份",
                         tooltip="保存最终模型前创建完整备份")
        self.components.switch(frame, 3, 1, ui_state, "backup_before_save")

        # save after
        self.components.label(frame, 4, 0, "保存间隔",
                         tooltip="训练时自动保存模型的间隔")
        self.components.time_entry(frame, 4, 1, ui_state, "save_every", "save_every_unit")

        # save now
        self.components.button(frame, 4, 3, "save now", self.save_now)

        # skip save
        self.components.label(frame, 5, 0, "跳过首个",
                         tooltip="Start saving automatically after this interval has elapsed")
        self.components.entry(frame, 5, 1, ui_state, "save_skip_first", width=50, sticky="nw")

        # save filename prefix
        self.components.label(frame, 6, 0, "保存文件名前缀",
                         tooltip="训练时保存模型的文件名前缀")
        self.components.entry(frame, 6, 1, ui_state, "save_filename_prefix")

    def build_embedding_tab_content(self, frame, controller, ui_state):
        # embedding model name
        self.components.label(frame, 0, 0, "基础嵌入",
                         tooltip="训练的基础嵌入，留空创建新嵌入")
        self.components.path_entry(
            frame, 0, 1, ui_state, "embedding.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # token count
        self.components.label(frame, 1, 0, "Token数",
                         tooltip="新嵌入的Token数，留空自动检测")
        self.components.entry(frame, 1, 1, ui_state, "embedding.token_count")

        # initial embedding text
        self.components.label(frame, 2, 0, "初始嵌入文本",
                         tooltip="创建新嵌入时的初始文本")
        self.components.entry(frame, 2, 1, ui_state, "embedding.initial_embedding_text")

        # embedding weight dtype
        self.components.label(frame, 3, 0, "嵌入权重数据类型",
                         tooltip="嵌入权重数据类型，可减少内存但降低精度")
        self.components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "embedding_weight_dtype")

        # placeholder
        self.components.label(frame, 4, 0, "占位符",
                         tooltip="在提示词中使用嵌入的占位符")
        self.components.entry(frame, 4, 1, ui_state, "embedding.placeholder")

        # output embedding
        self.components.label(frame, 5, 0, "输出嵌入",
                         tooltip="在文本编码器输出处计算嵌入，可改善大文本编码器效果并降低显存")
        self.components.switch(frame, 5, 1, ui_state, "embedding.is_output_embedding")

    def build_tools_tab_content(self, frame, controller, ui_state):
        # dataset
        self.components.label(frame, 0, 0, "数据集工具",
                         tooltip="打开标签工具")
        self.components.button(frame, 0, 1, "打开", self.open_dataset_tool)

        # video tools
        self.components.label(frame, 1, 0, "视频工具",
                         tooltip="打开视频工具")
        self.components.button(frame, 1, 1, "打开", self.open_video_tool)

        # convert model
        self.components.label(frame, 2, 0, "模型转换工具",
                         tooltip="打开模型转换工具")
        self.components.button(frame, 2, 1, "打开", self.open_convert_model_tool)

        # sample
        self.components.label(frame, 3, 0, "采样工具",
                         tooltip="打开模型采样工具")
        self.components.button(frame, 3, 1, "打开", self.open_sampling_tool)

        self.components.label(frame, 4, 0, "性能分析工具",
                         tooltip="打开性能分析工具")
        self.components.button(frame, 4, 1, "打开", self.open_profiling_tool)
