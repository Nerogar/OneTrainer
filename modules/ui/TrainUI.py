import ctypes
import datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog

import scripts.generate_debug_report
from modules.ui.AdditionalEmbeddingsTab import AdditionalEmbeddingsTab
from modules.ui.CaptionUI import CaptionUI
from modules.ui.CloudTab import CloudTab
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.LoraTab import LoraTab
from modules.ui.ModelTab import ModelTab
from modules.ui.ProfilingWindow import ProfilingWindow
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.ui.TrainingTab import TrainingTab
from modules.ui.VideoToolUI import VideoToolUI
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import torch

import customtkinter as ctk
from customtkinter import AppearanceModeTracker

# chunk for forcing Windows to ignore DPI scaling when moving between monitors
# fixes the long standing transparency bug https://github.com/Nerogar/OneTrainer/issues/90
if platform.system() == "Windows":
    with suppress(Exception):
        # https://learn.microsoft.com/en-us/windows/win32/hidpi/setting-the-default-dpi-awareness-for-a-process#setting-default-awareness-programmatically
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE

class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None
    training_callbacks: TrainCallbacks | None
    training_commands: TrainCommands | None

    _TRAIN_BUTTON_STYLES = {
        "idle": {
            "text": "Start Training",
            "state": "normal",
            "fg_color": "#198754",
            "hover_color": "#146c43",
            "text_color": "white",
            "text_color_disabled": "white",
        },
        "running": {
            "text": "Stop Training",
            "state": "normal",
            "fg_color": "#dc3545",
            "hover_color": "#bb2d3b",
            "text_color": "white",
        },
        "stopping": {
            "text": "Stopping...",
            "state": "disabled",
            "fg_color": "#dc3545",
            "hover_color": "#dc3545",
            "text_color": "white",
            "text_color_disabled": "white",
        },
    }

    def __init__(self):
        super().__init__()

        self.title("OneTrainer")
        self.geometry("1100x740")

        self.after(100, lambda: self._set_icon())

        # more efficient version of ctk.set_appearance_mode("System"), which retrieves the system theme on each main loop iteration
        ctk.set_appearance_mode("Light" if AppearanceModeTracker.detect_appearance_mode() == 0 else "Dark")
        ctk.set_default_color_theme("blue")

        self.train_config = TrainConfig.default_values()
        self.ui_state = UIState(self, self.train_config)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.eta_label = None
        self.training_button = None
        self.export_button = None
        self.tabview = None

        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.cloud_tab = None
        self.additional_embeddings_tab = None

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

        self.always_on_tensorboard_subprocess = None
        self.current_workspace_dir = self.train_config.workspace_dir
        self._check_start_always_on_tensorboard()

        self.workspace_dir_trace_id = self.ui_state.add_var_trace("workspace_dir", self._on_workspace_dir_change_trace)

        # Persistent profiling window.
        self.profiling_window = ProfilingWindow(self)

        self.protocol("WM_DELETE_WINDOW", self.__close)

    def __close(self):
        self.top_bar_component.save_default()
        self._stop_always_on_tensorboard()
        if hasattr(self, 'workspace_dir_trace_id'):
            self.ui_state.remove_var_trace("workspace_dir", self.workspace_dir_trace_id)
        self.quit()

    def top_bar(self, master):
        return TopBar(
            master,
            self.train_config,
            self.ui_state,
            self.change_model_type,
            self.change_training_method,
            self.load_preset,
        )

    def _set_icon(self):
        """Set the window icon safely after window is ready"""
        set_window_icon(self)

    def bottom_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        self.set_step_progress, self.set_epoch_progress = components.double_progress(frame, 0, 0, "step", "epoch")

        # status + ETA container
        self.status_frame = ctk.CTkFrame(frame, corner_radius=0, fg_color="transparent")
        self.status_frame.grid(row=0, column=1, sticky="w")
        self.status_frame.grid_rowconfigure(0, weight=0)
        self.status_frame.grid_rowconfigure(1, weight=0)
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = components.label(self.status_frame, 0, 0, "", pad=0,
                                             tooltip="Current status of the training run")
        self.eta_label = components.label(self.status_frame, 1, 0, "", pad=0)

        # padding
        frame.grid_columnconfigure(2, weight=1)


        # export button
        self.export_button = components.button(frame, 0, 3, "Export", self.export_training,
                                             width=60, padx=5, pady=(15, 0),
                                             tooltip="Export the current configuration as a script to run without a UI")

        # debug button
        components.button(frame, 0, 4, "Debug", self.generate_debug_package,
                                       width=60, padx=(5, 25), pady=(15, 0),
                                       tooltip="Generate a zip file with config.json, debug_report.log and settings diff, use this to report bugs or issues")

        # tensorboard button
        components.button(frame, 0, 5, "Tensorboard", self.open_tensorboard,
                                             width=100, padx=(0, 5), pady=(15, 0))

        # training button
        self.training_button = components.button(frame, 0, 6, "Start Training", self.start_training,
                                                 padx=(5, 20), pady=(15, 0))
        self._set_training_button_style("idle")  # centralized styling

        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(frame)
        self.tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab = self.create_general_tab(self.tabview.add("general"))
        self.model_tab = self.create_model_tab(self.tabview.add("model"))
        self.data_tab = self.create_data_tab(self.tabview.add("data"))
        self.concepts_tab = self.create_concepts_tab(self.tabview.add("concepts"))
        self.training_tab = self.create_training_tab(self.tabview.add("training"))
        self.sampling_tab = self.create_sampling_tab(self.tabview.add("sampling"))
        self.backup_tab = self.create_backup_tab(self.tabview.add("backup"))
        self.tools_tab = self.create_tools_tab(self.tabview.add("tools"))
        self.additional_embeddings_tab = self.create_additional_embeddings_tab(self.tabview.add("additional embeddings"))
        self.cloud_tab = self.create_cloud_tab(self.tabview.add("cloud"))

        self.change_training_method(self.train_config.training_method)

        return frame

    def create_general_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)
        frame.grid_columnconfigure(3, weight=1)

        # workspace dir
        components.label(frame, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run are saved")
        components.dir_entry(frame, 0, 1, self.ui_state, "workspace_dir", command=self._on_workspace_dir_change)

        # cache dir
        components.label(frame, 0, 2, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        components.dir_entry(frame, 0, 3, self.ui_state, "cache_dir")

        # continue from previous backup
        components.label(frame, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/backup")
        components.switch(frame, 2, 1, self.ui_state, "continue_last_backup")

        # only cache
        components.label(frame, 2, 2, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        components.switch(frame, 2, 3, self.ui_state, "only_cache")

        # debug
        components.label(frame, 4, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        components.switch(frame, 4, 1, self.ui_state, "debug_mode")

        components.label(frame, 4, 2, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        components.dir_entry(frame, 4, 3, self.ui_state, "debug_dir")

        # tensorboard
        components.label(frame, 6, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        components.switch(frame, 6, 1, self.ui_state, "tensorboard")

        components.label(frame, 6, 2, "Always-On Tensorboard",
                         tooltip="Keep Tensorboard accessible even when not training. Useful for monitoring completed training sessions.")
        components.switch(frame, 6, 3, self.ui_state, "tensorboard_always_on", command=self._on_always_on_tensorboard_toggle)

        components.label(frame, 7, 0, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")
        components.switch(frame, 7, 1, self.ui_state, "tensorboard_expose")
        components.label(frame, 7, 2, "Tensorboard Port",
                         tooltip="Port to use for Tensorboard link")
        components.entry(frame, 7, 3, self.ui_state, "tensorboard_port")


        # validation
        components.label(frame, 8, 0, "Validation",
                         tooltip="Enable validation steps and add new graph in tensorboard")
        components.switch(frame, 8, 1, self.ui_state, "validation")

        components.label(frame, 8, 2, "Validate after",
                         tooltip="The interval used when validate training")
        components.time_entry(frame, 8, 3, self.ui_state, "validate_after", "validate_after_unit")

        # device
        components.label(frame, 10, 0, "Dataloader Threads",
                         tooltip="Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.")
        components.entry(frame, 10, 1, self.ui_state, "dataloader_threads")

        components.label(frame, 11, 0, "Train Device",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\". Must be \"cuda\" for multi-GPU training.")
        components.entry(frame, 11, 1, self.ui_state, "train_device")

        components.label(frame, 12, 0, "Multi-GPU",
                         tooltip="Enable multi-GPU training")
        components.switch(frame, 12, 1, self.ui_state, "multi_gpu")
        components.label(frame, 12, 2, "Device Indexes",
                         tooltip="Multi-GPU: A comma-separated list of device indexes. If empty, all your GPUs are used. With a list such as \"0,1,3,4\" you can omit a GPU, for example an on-board graphics GPU.")
        components.entry(frame, 12, 3, self.ui_state, "device_indexes")

        components.label(frame, 13, 0, "Gradient Reduce Precision",
                         tooltip="WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
                                 "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
                                 "FLOAT_32: Reduce gradients in float32\n"
                                 "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
                         wide_tooltip=True)
        components.options(frame, 13, 1, [str(x) for x in list(GradientReducePrecision)], self.ui_state,
                           "gradient_reduce_precision")

        components.label(frame, 13, 2, "Fused Gradient Reduce",
                         tooltip="Multi-GPU: Gradient synchronisation during the backward pass. Can be more efficient, especially with Async Gradient Reduce")
        components.switch(frame, 13, 3, self.ui_state, "fused_gradient_reduce")

        components.label(frame, 14, 0, "Async Gradient Reduce",
                         tooltip="Multi-GPU: Asynchroniously start the gradient reduce operations during the backward pass. Can be more efficient, but requires some VRAM.")
        components.switch(frame, 14, 1, self.ui_state, "async_gradient_reduce")
        components.label(frame, 14, 2, "Buffer size (MB)",
                         tooltip="Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"")
        components.entry(frame, 14, 3, self.ui_state, "async_gradient_reduce_buffer")

        components.label(frame, 15, 0, "Temp Device",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")
        components.entry(frame, 15, 1, self.ui_state, "temp_device")

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
        components.label(frame, 0, 0, "Aspect Ratio Bucketing",
                         tooltip="Aspect ratio bucketing enables training on images with different aspect ratios")
        components.switch(frame, 0, 1, self.ui_state, "aspect_ratio_bucketing")

        # latent caching
        components.label(frame, 1, 0, "Latent Caching",
                         tooltip="Caching of intermediate training data that can be re-used between epochs")
        components.switch(frame, 1, 1, self.ui_state, "latent_caching")

        # clear cache before training
        components.label(frame, 2, 0, "Clear cache before training",
                         tooltip="Clears the cache directory before starting to train. Only disable this if you want to continue using the same cached data. Disabling this can lead to errors, if other settings are changed during a restart")
        components.switch(frame, 2, 1, self.ui_state, "clear_cache_before_training")

        frame.pack(fill="both", expand=1)
        return frame

    def create_concepts_tab(self, master):
        return ConceptTab(master, self.train_config, self.ui_state)

    def create_training_tab(self, master) -> TrainingTab:
        return TrainingTab(master, self.train_config, self.ui_state)

    def create_cloud_tab(self, master) -> CloudTab:
        return CloudTab(master, self.train_config, self.ui_state,parent=self)

    def create_sampling_tab(self, master):
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # sample after
        top_frame = ctk.CTkFrame(master=master, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")
        sub_frame = ctk.CTkFrame(master=top_frame, corner_radius=0, fg_color="transparent")
        sub_frame.grid(row=1, column=0, sticky="nsew", columnspan=6)

        components.label(top_frame, 0, 0, "Sample After",
                         tooltip="The interval used when automatically sampling from the model during training")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")

        components.label(top_frame, 0, 2, "Skip First",
                         tooltip="Start sampling automatically after this interval has elapsed.")
        components.entry(top_frame, 0, 3, self.ui_state, "sample_skip_first", width=50, sticky="nw")

        components.label(top_frame, 0, 4, "Format",
                         tooltip="File Format used when saving samples")
        components.options_kv(top_frame, 0, 5, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], self.ui_state, "sample_image_format")

        components.button(top_frame, 0, 6, "sample now", self.sample_now)

        components.button(top_frame, 0, 7, "manual sample", self.open_sample_ui)

        components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                         tooltip="Whether to include non-ema sampling when using ema.")
        components.switch(sub_frame, 0, 1, self.ui_state, "non_ema_sampling")

        components.label(sub_frame, 0, 2, "Samples to Tensorboard",
                         tooltip="Whether to include sample images in the Tensorboard output.")
        components.switch(sub_frame, 0, 3, self.ui_state, "samples_to_tensorboard")

        # table
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        return SamplingTab(frame, self.train_config, self.ui_state)

    def create_backup_tab(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=50)
        frame.grid_columnconfigure(3, weight=0)
        frame.grid_columnconfigure(4, weight=1)

        # backup after
        components.label(frame, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        components.time_entry(frame, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        # backup now
        components.button(frame, 0, 3, "backup now", self.backup_now)

        # rolling backup
        components.label(frame, 1, 0, "Rolling Backup",
                         tooltip="If rolling backups are enabled, older backups are deleted automatically")
        components.switch(frame, 1, 1, self.ui_state, "rolling_backup")

        # rolling backup count
        components.label(frame, 1, 3, "Rolling Backup Count",
                         tooltip="Defines the number of backups to keep if rolling backups are enabled")
        components.entry(frame, 1, 4, self.ui_state, "rolling_backup_count")

        # backup before save
        components.label(frame, 2, 0, "Backup Before Save",
                         tooltip="Create a full backup before saving the final model")
        components.switch(frame, 2, 1, self.ui_state, "backup_before_save")

        # save after
        components.label(frame, 3, 0, "Save Every",
                         tooltip="The interval used when automatically saving the model during training")
        components.time_entry(frame, 3, 1, self.ui_state, "save_every", "save_every_unit")

        # save now
        components.button(frame, 3, 3, "save now", self.save_now)

        # skip save
        components.label(frame, 4, 0, "Skip First",
                         tooltip="Start saving automatically after this interval has elapsed")
        components.entry(frame, 4, 1, self.ui_state, "save_skip_first", width=50, sticky="nw")

        # save filename prefix
        components.label(frame, 5, 0, "Save Filename Prefix",
                         tooltip="The prefix for filenames used when saving the model during training")
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
        components.label(frame, 0, 0, "LoRA base model",
                         tooltip="The base LoRA to train on. Leave empty to create a new LoRA")
        components.file_entry(
            frame, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # lora rank
        components.label(frame, 1, 0, "LoRA rank",
                         tooltip="The rank parameter used when creating a new LoRA")
        components.entry(frame, 1, 1, self.ui_state, "lora_rank")

        # lora rank
        components.label(frame, 2, 0, "LoRA alpha",
                         tooltip="The alpha parameter used when creating a new LoRA")
        components.entry(frame, 2, 1, self.ui_state, "lora_alpha")

        # Dropout Percentage
        components.label(frame, 3, 0, "Dropout Probability",
                         tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
        components.entry(frame, 3, 1, self.ui_state, "dropout_probability")

        # lora weight dtype
        components.label(frame, 4, 0, "LoRA Weight Data Type",
                         tooltip="The LoRA weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(frame, 4, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        # For use with additional embeddings.
        components.label(frame, 5, 0, "Bundle Embeddings",
                         tooltip="Bundles any additional embeddings into the LoRA output file, rather than as separate files")
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
        components.label(frame, 0, 0, "Base embedding",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        components.file_entry(
            frame, 0, 1, self.ui_state, "embedding.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # token count
        components.label(frame, 1, 0, "Token count",
                         tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        components.entry(frame, 1, 1, self.ui_state, "embedding.token_count")

        # initial embedding text
        components.label(frame, 2, 0, "Initial embedding text",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(frame, 2, 1, self.ui_state, "embedding.initial_embedding_text")

        # embedding weight dtype
        components.label(frame, 3, 0, "Embedding Weight Data Type",
                         tooltip="The Embedding weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "embedding_weight_dtype")

        # placeholder
        components.label(frame, 4, 0, "Placeholder",
                         tooltip="The placeholder used when using the embedding in a prompt")
        components.entry(frame, 4, 1, self.ui_state, "embedding.placeholder")

        # output embedding
        components.label(frame, 5, 0, "Output embedding",
                         tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        components.switch(frame, 5, 1, self.ui_state, "embedding.is_output_embedding")

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
        components.label(frame, 0, 0, "Dataset Tools",
                         tooltip="Open the captioning tool")
        components.button(frame, 0, 1, "Open", self.open_dataset_tool)

        # video tools
        components.label(frame, 1, 0, "Video Tools",
                         tooltip="Open the video tools")
        components.button(frame, 1, 1, "Open", self.open_video_tool)

        # convert model
        components.label(frame, 2, 0, "Convert Model Tools",
                         tooltip="Open the model conversion tool")
        components.button(frame, 2, 1, "Open", self.open_convert_model_tool)

        # sample
        components.label(frame, 3, 0, "Sampling Tool",
                         tooltip="Open the model sampling tool")
        components.button(frame, 3, 1, "Open", self.open_sampling_tool)

        components.label(frame, 4, 0, "Profiling Tool",
                         tooltip="Open the profiling tools.")
        components.button(frame, 4, 1, "Open", self.open_profiling_tool)

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
        webbrowser.open("http://localhost:" + str(self.train_config.tensorboard_port), new=0, autoraise=False)

    def _calculate_eta_string(self, train_progress: TrainProgress, max_step: int, max_epoch: int) -> str | None:
        spent_total = time.monotonic() - self.start_time
        steps_done = train_progress.epoch * max_step + train_progress.epoch_step
        remaining_steps = (max_epoch - train_progress.epoch - 1) * max_step + (max_step - train_progress.epoch_step)
        total_eta = spent_total / steps_done * remaining_steps

        if train_progress.global_step <= 30:
            return "Estimating ..."

        td = datetime.timedelta(seconds=total_eta)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def set_eta_label(self, train_progress: TrainProgress, max_step: int, max_epoch: int):
        eta_str = self._calculate_eta_string(train_progress, max_step, max_epoch)
        if eta_str is not None:
            self.eta_label.configure(text=f"ETA: {eta_str}")
        else:
            self.eta_label.configure(text="")

    def delete_eta_label(self):
        self.eta_label.configure(text="")

    def on_update_train_progress(self, train_progress: TrainProgress, max_step: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_step)
        self.set_epoch_progress(train_progress.epoch, max_epoch)
        self.set_eta_label(train_progress, max_step, max_epoch)

    def on_update_status(self, status: str):
        self.status_label.configure(text=status)

    def open_dataset_tool(self):
        window = CaptionUI(self, None, False)
        self.wait_window(window)

    def open_video_tool(self):
        window = VideoToolUI(self)
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

    def generate_debug_package(self):
        zip_path = filedialog.askdirectory(
            initialdir=".",
            title="Select Directory to Save Debug Package"
        )

        if not zip_path:
            return

        zip_path = Path(zip_path) / "OneTrainer_debug_report.zip"

        self.on_update_status("Generating debug package...")

        try:
            config_json_string = json.dumps(self.train_config.to_pack_dict(secrets=False))
            scripts.generate_debug_report.create_debug_package(str(zip_path), config_json_string)
            self.on_update_status(f"Debug package saved to {zip_path.name}")
        except Exception as e:
            traceback.print_exc()
            self.on_update_status(f"Error generating debug package: {e}")


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

        trainer = create.create_trainer(self.train_config, self.training_callbacks, self.training_commands, reattach=self.cloud_tab.reattach)
        try:
            trainer.start()
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)

            self.start_time = time.monotonic()
            trainer.train()
        except Exception:
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
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
            self.on_update_status("Error: check the console for details")
        else:
            self.on_update_status("Stopped")
        self.delete_eta_label()

        # queue UI update on Tk main thread; _set_training_button_idle applies shared styles, avoid potential race/crash
        self.after(0, self._set_training_button_idle)

        if self.train_config.tensorboard_always_on and not self.always_on_tensorboard_subprocess:
            self.after(0, self._start_always_on_tensorboard)

    def start_training(self):
        if self.training_thread is None:
            self.save_default()
            self._set_training_button_running()

            if self.train_config.tensorboard and not self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._stop_always_on_tensorboard()

            self.training_commands = TrainCommands()
            torch_gc()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            self._set_training_button_stopping()
            self.on_update_status("Stopping ...")
            self.training_commands.stop()

    def save_default(self):
        self.top_bar_component.save_default()
        self.concepts_tab.save_current_config()
        self.sampling_tab.save_current_config()
        self.additional_embeddings_tab.save_current_config()

    def export_training(self):
        file_path = filedialog.asksaveasfilename(filetypes=[
            ("All Files", "*.*"),
            ("json", "*.json"),
        ], initialdir=".", initialfile="config.json")

        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.train_config.to_pack_dict(secrets=False), f, indent=4)

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

    def _check_start_always_on_tensorboard(self):
        if self.train_config.tensorboard_always_on and not self.always_on_tensorboard_subprocess:
            self._start_always_on_tensorboard()

    def _start_always_on_tensorboard(self):
        if self.always_on_tensorboard_subprocess:
            self._stop_always_on_tensorboard()

        tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        tensorboard_log_dir = os.path.join(self.train_config.workspace_dir, "tensorboard")

        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)

        tensorboard_args = [
            tensorboard_executable,
            "--logdir",
            tensorboard_log_dir,
            "--port",
            str(self.train_config.tensorboard_port),
            "--samples_per_plugin=images=100,scalars=10000",
        ]

        if self.train_config.tensorboard_expose:
            tensorboard_args.append("--bind_all")

        try:
            self.always_on_tensorboard_subprocess = subprocess.Popen(tensorboard_args)
        except Exception:
            self.always_on_tensorboard_subprocess = None

    def _stop_always_on_tensorboard(self):
        if self.always_on_tensorboard_subprocess:
            try:
                self.always_on_tensorboard_subprocess.terminate()
                self.always_on_tensorboard_subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.always_on_tensorboard_subprocess.kill()
            except Exception:
                pass
            finally:
                self.always_on_tensorboard_subprocess = None

    def _on_workspace_dir_change(self, new_workspace_dir: str):
        if new_workspace_dir != self.current_workspace_dir:
            self.current_workspace_dir = new_workspace_dir

            if self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._start_always_on_tensorboard()

    def _on_workspace_dir_change_trace(self, *args):
        new_workspace_dir = self.train_config.workspace_dir
        if new_workspace_dir != self.current_workspace_dir:
            self.current_workspace_dir = new_workspace_dir

            if self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._start_always_on_tensorboard()

    def _on_always_on_tensorboard_toggle(self):
        if self.train_config.tensorboard_always_on:
            if not (self.training_thread and self.train_config.tensorboard):
                self._start_always_on_tensorboard()
        else:
            if not (self.training_thread and self.train_config.tensorboard):
                self._stop_always_on_tensorboard()

    def _set_training_button_style(self, mode: str):
        if not self.training_button:
            return
        style = self._TRAIN_BUTTON_STYLES.get(mode)
        if not style:
            return
        self.training_button.configure(**style)

    def _set_training_button_idle(self):
        self._set_training_button_style("idle")

    def _set_training_button_running(self):
        self._set_training_button_style("running")

    def _set_training_button_stopping(self):
        self._set_training_button_style("stopping")
