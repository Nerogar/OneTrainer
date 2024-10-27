import json
import threading
import traceback
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog

from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.AdditionalEmbeddingsTab import AdditionalEmbeddingsTab
from modules.ui.CaptionUI import CaptionUI
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.LoraTab import LoraTab
from modules.ui.ModelTab import ModelTab
from modules.ui.ProfilingWindow import ProfilingWindow
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.ui.TrainingTab import TrainingTab
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.zluda import ZLUDA

import torch

import customtkinter as ctk


class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None
    training_callbacks: TrainCallbacks | None
    training_commands: TrainCommands | None

    def __init__(self):
        super().__init__()

        self.title("OneTrainer")
        self.geometry("1100x740")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.train_config = TrainConfig.default_values()
        self.ui_state = UIState(self, self.train_config)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.training_button = None
        self.export_button = None
        self.tabview = None

        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.additional_embeddings_tab = None

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

        # Persistent profiling window.
        self.profiling_window = ProfilingWindow(self)

    def close(self):
        self.top_bar_component.save_default()

    def top_bar(self, master):
        return TopBar(
            master,
            self.train_config,
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def bottom_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        self.set_step_progress, self.set_epoch_progress = components.double_progress(frame, 0, 0, "步数", "轮次")

        self.status_label = components.label(frame, 0, 1, "",
                                             tooltip="训练运行的当前状态")

        # padding
        frame.grid_columnconfigure(2, weight=1)

        # tensorboard button
        components.button(frame, 0, 3, "Tensorboard日志", self.open_tensorboard)

        # training button
        self.training_button = components.button(frame, 0, 4, "开始训练", self.start_training)

        # export button
        self.export_button = components.button(frame, 0, 5, "导出", self.export_training,
                                               tooltip="将当前配置导出为脚本，以便在没有 UI 的情况下运行。")


        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.create_general_tab(self.tabview.add("基础设置"))
        self.model_tab = self.create_model_tab(self.tabview.add("模型"))
        self.data_tab = self.create_data_tab(self.tabview.add("数据"))
        self.create_concepts_tab(self.tabview.add("概念"))
        self.training_tab = self.create_training_tab(self.tabview.add("训练参数"))
        self.create_sampling_tab(self.tabview.add("采样图"))
        self.backup_tab = self.create_backup_tab(self.tabview.add("备份"))
        self.tools_tab = self.create_tools_tab(self.tabview.add("工具"))
        self.additional_embeddings_tab = self.create_additional_embeddings_tab(self.tabview.add("附加嵌入"))

        self.change_training_method(self.train_config.training_method)

        return frame

    def create_general_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # workspace dir
        components.label(frame, 0, 0, "工作区目录",
                         tooltip="保存此次训练运行的所有文件的目录。")
        components.dir_entry(frame, 0, 1, self.ui_state, "workspace_dir")

        # cache dir
        components.label(frame, 1, 0, "缓存目录",
                         tooltip="保存缓存数据的目录。")
        components.dir_entry(frame, 1, 1, self.ui_state, "cache_dir")

        # continue from previous backup
        components.label(frame, 2, 0, "从上次备份继续",
                         tooltip="自动从<workspace>/backup中保存的上次备份继续训练。")
        components.switch(frame, 2, 1, self.ui_state, "continue_last_backup")

        # only cache
        components.label(frame, 3, 0, "仅缓存",
                         tooltip="仅填充缓存，不进行任何训练。")
        components.switch(frame, 3, 1, self.ui_state, "only_cache")

        # debug
        components.label(frame, 4, 0, "调试模式",
                         tooltip="在训练过程中将调试信息保存到调试目录。")
        components.switch(frame, 4, 1, self.ui_state, "debug_mode")

        components.label(frame, 5, 0, "调试目录",
                         tooltip="保存调试数据的目录。")
        components.dir_entry(frame, 5, 1, self.ui_state, "debug_dir")

        # tensorboard
        components.label(frame, 6, 0, "Tensorboard日志",
                         tooltip="在训练过程中启动 Tensorboard Web UI。")
        components.switch(frame, 6, 1, self.ui_state, "tensorboard")

        components.label(frame, 7, 0, "公开 Tensorboard",
                         tooltip="将 Tensorboard Web UI 公开给所有网络接口（使其可从网络访问）。")
        components.switch(frame, 7, 1, self.ui_state, "tensorboard_expose")

        # validation
        components.label(frame, 8, 0, "验证",
                         tooltip="启用验证步骤并在 Tensorboard 中添加新的图形。")
        components.switch(frame, 8, 1, self.ui_state, "validation")

        components.label(frame, 9, 0, "验证间隔",
                         tooltip="验证训练时使用的间隔。")
        components.time_entry(frame, 9, 1, self.ui_state, "validate_after", "validate_after_unit")

        # device
        components.label(frame, 10, 0, "数据加载器线程",
                         tooltip="数据加载器使用的线程数。如果 GPU 在缓存期间有空间，则增加线程数；如果 GPU 在缓存期间内存不足，则减少线程数。")
        components.entry(frame, 10, 1, self.ui_state, "dataloader_threads")

        components.label(frame, 11, 0, "训练设备",
                         tooltip="用于训练的设备。可以是“cuda”、“cuda:0”、“cuda:1”等。默认值为“cuda”。")
        components.entry(frame, 11, 1, self.ui_state, "train_device")

        components.label(frame, 12, 0, "临时设备",
                         tooltip="用于在模型未使用时暂时卸载模型的设备。默认值为“cpu”。")
        components.entry(frame, 12, 1, self.ui_state, "temp_device")

        frame.pack(fill="both", expand=1)
        return frame

    def create_model_tab(self, master):
        return ModelTab(master, self.train_config, self.ui_state)

    def create_data_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # aspect ratio bucketing
        components.label(frame, 0, 0, "长宽比分桶",
                         tooltip="长宽比分桶允许对不同长宽比的图像进行训练。")
        components.switch(frame, 0, 1, self.ui_state, "aspect_ratio_bucketing")

        # latent caching
        components.label(frame, 1, 0, "潜在缓存",
                         tooltip="缓存中间训练数据，这些数据可以在不同轮次之间重复使用。")
        components.switch(frame, 1, 1, self.ui_state, "latent_caching")

        # clear cache before training
        components.label(frame, 2, 0, "训练前清除缓存",
                         tooltip="在开始训练之前清除缓存目录。只有在您想继续使用相同的缓存数据时才禁用此功能。如果在重启期间更改了其他设置，禁用此功能可能会导致错误。")
        components.switch(frame, 2, 1, self.ui_state, "clear_cache_before_training")

        frame.pack(fill="both", expand=1)
        return frame

    def create_concepts_tab(self, master):
        ConceptTab(master, self.train_config, self.ui_state)

    def create_training_tab(self, master) -> TrainingTab:
        return TrainingTab(master, self.train_config, self.ui_state)

    def create_sampling_tab(self, master):
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # sample after
        top_frame = ctk.CTkFrame(master=master, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")
        sub_frame = ctk.CTkFrame(master=top_frame, corner_radius=0, fg_color="transparent")
        sub_frame.grid(row=1, column=0, sticky="nsew", columnspan=6)
        components.label(top_frame, 0, 0, "采样间隔",
                         tooltip="在训练期间从模型自动采样时使用的间隔。")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")

        components.label(top_frame, 0, 2, "格式",
                         tooltip="保存样本时使用的文件格式。")
        components.options_kv(top_frame, 0, 3, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], self.ui_state, "sample_image_format")

        components.button(top_frame, 0, 4, "立即采样", self.sample_now)

        components.button(top_frame, 0, 5, "手动采样", self.open_sample_ui)

        components.label(sub_frame, 0, 0, "非 EMA 采样",
                         tooltip="在使用 EMA 时是否包含非 EMA 采样。")
        components.switch(sub_frame, 0, 1, self.ui_state, "non_ema_sampling")

        components.label(sub_frame, 0, 2, "Tensorboard 采样",
                         tooltip="是否将样本图像包含在 Tensorboard 输出中。")
        components.switch(sub_frame, 0, 3, self.ui_state, "samples_to_tensorboard")

        # table
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        SamplingTab(frame, self.train_config, self.ui_state)

    def create_backup_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # backup after
        components.label(frame, 0, 0, "备份间隔",
                         tooltip="在训练期间自动创建模型备份时使用的间隔。")
        components.time_entry(frame, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        # backup now
        components.button(frame, 0, 3, "立即备份", self.backup_now)

        # rolling backup
        components.label(frame, 1, 0, "滚动备份",
                         tooltip="如果启用了滚动备份，则会自动删除旧的备份。")
        components.switch(frame, 1, 1, self.ui_state, "rolling_backup")

        # rolling backup count
        components.label(frame, 1, 3, "滚动备份数量",
                         tooltip="如果启用了滚动备份，则定义要保留的备份数量。")
        components.entry(frame, 1, 4, self.ui_state, "rolling_backup_count")

        # backup before save
        components.label(frame, 2, 0, "保存前备份",
                         tooltip="在保存最终模型之前创建完整备份。")
        components.switch(frame, 2, 1, self.ui_state, "backup_before_save")

        # save after
        components.label(frame, 3, 0, "保存间隔",
                         tooltip="在训练期间自动保存模型时使用的间隔。")
        components.time_entry(frame, 3, 1, self.ui_state, "save_every", "save_every_unit")

        # save now
        components.button(frame, 3, 3, "立即保存", self.save_now)

        # skip save
        components.label(frame, 4, 0, "跳过第一个",
                         tooltip="在该间隔过去后开始自动保存。")
        components.entry(frame, 4, 1, self.ui_state, "save_skip_first", width=50, sticky="nw")

        # save filename prefix
        components.label(frame, 5, 0, "保存文件名前缀",
                         tooltip="在训练期间保存模型时使用的文件名前缀。")
        components.entry(frame, 5, 1, self.ui_state, "save_filename_prefix")

        frame.pack(fill="both", expand=1)
        return frame

    def lora_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # lora model name
        components.label(frame, 0, 0, "LoRA 基础模型",
                         tooltip="要训练的 LoRA 基础模型。留空以创建新的 LoRA。")
        components.file_entry(
            frame, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # lora rank
        components.label(frame, 1, 0, "LoRA 秩",
                         tooltip="创建新的 LoRA 时使用的秩参数。")
        components.entry(frame, 1, 1, self.ui_state, "lora_rank")

        # lora rank
        components.label(frame, 2, 0, "LoRA alpha",
                         tooltip="创建新的 LoRA 时使用的 alpha 参数。")
        components.entry(frame, 2, 1, self.ui_state, "lora_alpha")

        # Dropout Percentage
        components.label(frame, 3, 0, "丢弃概率",
                         tooltip="丢弃概率。此百分比的模型节点将在每个训练步骤中被随机忽略。有助于防止过拟合。范围0-1，为0则不启用")
        components.entry(frame, 3, 1, self.ui_state, "dropout_probability")

        # lora weight dtype
        components.label(frame, 4, 0, "LoRA 权重数据类型",
                         tooltip="用于训练的 LoRA 权重数据类型。这可以减少内存消耗，但会降低精度。")
        components.options_kv(frame, 4, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        # For use with additional embeddings.
        components.label(frame, 5, 0, "捆绑嵌入",
                         tooltip="将任何额外的嵌入捆绑到 LoRA 输出文件中，而不是作为单独的文件。")
        components.switch(frame, 5, 1, self.ui_state, "bundle_additional_embeddings")

        frame.pack(fill="both", expand=1)
        return frame

    def embedding_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # embedding model name
        components.label(frame, 0, 0, "基础嵌入",
                         tooltip="要训练的基础嵌入。留空以创建新的嵌入。")
        components.file_entry(
            frame, 0, 1, self.ui_state, "embedding.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # token count
        components.label(frame, 1, 0, "令牌数量",
                         tooltip="创建新的嵌入时使用的令牌数量。")
        components.entry(frame, 1, 1, self.ui_state, "embedding.token_count")

        # initial embedding text
        components.label(frame, 2, 0, "初始嵌入文本",
                         tooltip="创建新的嵌入时使用的初始嵌入文本。")
        components.entry(frame, 2, 1, self.ui_state, "embedding.initial_embedding_text")

        # embedding weight dtype
        components.label(frame, 3, 0, "嵌入权重数据类型",
                         tooltip="用于训练的嵌入权重数据类型。这可以减少内存消耗，但会降低精度。")
        components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "embedding_weight_dtype")

        # placeholder
        components.label(frame, 4, 0, "触发词",
                         tooltip="在提示中使用嵌入时使用的触发词。")
        components.entry(frame, 4, 1, self.ui_state, "embedding.placeholder")

        frame.pack(fill="both", expand=1)
        return frame

    def create_additional_embeddings_tab(self, master):
        return AdditionalEmbeddingsTab(master, self.train_config, self.ui_state)

    def create_tools_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # dataset
        components.label(frame, 0, 0, "数据集工具",
                         tooltip="打开标注工具")
        components.button(frame, 0, 1, "打开", self.open_dataset_tool)

        # convert model
        components.label(frame, 1, 0, "模型转换工具",
                         tooltip="打开模型转换工具")
        components.button(frame, 1, 1, "打开", self.open_convert_model_tool)

        # sample
        components.label(frame, 2, 0, "采样工具",
                         tooltip="打开模型采样工具")
        components.button(frame, 2, 1, "打开", self.open_sampling_tool)

        components.label(frame, 3, 0, "性能分析工具",
                         tooltip="打开性能分析工具。")
        components.button(frame, 3, 1, "打开", self.open_profiling_tool)

        frame.pack(fill="both", expand=1)
        return frame

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()

        if self.training_tab:
            self.training_tab.refresh_ui()

        if self.lora_tab:
            self.lora_tab.refresh_ui()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        if training_method != TrainingMethod.LORA and "LoRA" in self.tabview._tab_dict:
            self.tabview.delete("LoRA")
            self.lora_tab = None
        if training_method != TrainingMethod.EMBEDDING and "embedding" in self.tabview._tab_dict:
            self.tabview.delete("embedding")

        if training_method == TrainingMethod.LORA and "LoRA" not in self.tabview._tab_dict:
            self.lora_tab = LoraTab(self.tabview.add("LoRA"), self.train_config, self.ui_state)
        if training_method == TrainingMethod.EMBEDDING and "embedding" not in self.tabview._tab_dict:
            self.embedding_tab(self.tabview.add("embedding"))

    def load_preset(self):
        if not self.tabview:
            return

        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()

    def open_tensorboard(self):
        webbrowser.open("http://localhost:6006/", new=0, autoraise=False)

    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)

    def on_update_status(self, status: str):
        self.status_label.configure(text=status)

    def open_dataset_tool(self):
        window = CaptionUI(self, None, False)
        self.wait_window(window)

    def open_convert_model_tool(self):
        window = ConvertModelUI(self)
        self.wait_window(window)

    def open_sampling_tool(self):
        if not self.training_callbacks and not self.training_commands:
            window = SampleWindow(
                self,
                train_config=self.train_config,
            )
            self.wait_window(window)
            torch_gc()

    def open_profiling_tool(self):
        self.profiling_window.deiconify()

    def open_sample_ui(self):
        training_callbacks = self.training_callbacks
        training_commands = self.training_commands

        if training_callbacks and training_commands:
            window = SampleWindow(
                self,
                callbacks=training_callbacks,
                commands=training_commands,
            )
            self.wait_window(window)
            training_callbacks.set_on_sample_custom()

    def __training_thread_function(self):
        error_caught = False

        self.training_callbacks = TrainCallbacks(
            on_update_train_progress=self.on_update_train_progress,
            on_update_status=self.on_update_status,
        )

        ZLUDA.initialize_devices(self.train_config)

        trainer = GenericTrainer(self.train_config, self.training_callbacks, self.training_commands)

        try:
            trainer.start()
            trainer.train()
        except Exception:
            error_caught = True
            traceback.print_exc()

        trainer.end()

        # clear gpu memory
        del trainer
        self.training_thread = None
        self.training_commands = None
        torch.clear_autocast_cache()
        torch_gc()

        if error_caught:
            self.on_update_status("错误：查看控制台获取更多信息")
        else:
            self.on_update_status("已停止")

        self.training_button.configure(text="开始训练", state="normal")

    def start_training(self):
        if self.training_thread is None:
            self.top_bar_component.save_default()

            self.training_button.configure(text="停止训练", state="normal")

            self.training_commands = TrainCommands()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            self.training_button.configure(state="disabled")
            self.on_update_status("正在停止")
            self.training_commands.stop()

    def export_training(self):
        file_path = filedialog.asksaveasfilename(filetypes=[
            ("所有文件", "*.*"),
            ("json", "*.json"),
        ], initialdir=".", initialfile="config.json")

        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.train_config.to_pack_dict(), f, indent=4)

    def sample_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.sample_default()

    def backup_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.backup()

    def save_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.save()
