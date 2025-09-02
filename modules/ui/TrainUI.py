import ctypes
import json
import platform
import threading
import traceback
import webbrowser
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog

from modules.trainer.CloudTrainer import CloudTrainer
from modules.trainer.GenericTrainer import GenericTrainer
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
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TensorboardMode import TensorboardMode
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.tensorboard_util import start_filtered_tensorboard, stop_tensorboard
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState
from modules.zluda import ZLUDA

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
        self._initialize_tensorboard()

        self.workspace_dir_trace_id = self.ui_state.add_var_trace("workspace_dir", self._on_workspace_dir_change_trace)
        self.tensorboard_mode_trace_id = self.ui_state.add_var_trace("tensorboard_mode", self._on_tensorboard_mode_change)

        # Persistent profiling window.
        self.profiling_window = ProfilingWindow(self)

        self.protocol("WM_DELETE_WINDOW", self.__close)

    def __close(self):
        self.top_bar_component.save_default()
        self._stop_always_on_tensorboard()
        if hasattr(self, 'workspace_dir_trace_id'):
            self.ui_state.remove_var_trace("workspace_dir", self.workspace_dir_trace_id)
        if hasattr(self, 'tensorboard_mode_trace_id'):
            self.ui_state.remove_var_trace("tensorboard_mode", self.tensorboard_mode_trace_id)
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

        self.status_label = components.label(frame, 0, 1, "",
                                             tooltip="Current status of the training run")

        # padding
        frame.grid_columnconfigure(2, weight=1)

        # tensorboard button
        components.button(frame, 0, 3, "Tensorboard", self.open_tensorboard)

        # training button
        self.training_button = components.button(frame, 0, 4, "Start Training", self.start_training)

        # export button
        self.export_button = components.button(frame, 0, 5, "Export", self.export_training,
                                               tooltip="Export the current configuration as a script to run without a UI")


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
        components.label(frame, 1, 0, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        components.dir_entry(frame, 1, 1, self.ui_state, "cache_dir")

        # continue from previous backup
        components.label(frame, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/backup")
        components.switch(frame, 2, 1, self.ui_state, "continue_last_backup")

        # only cache
        components.label(frame, 3, 0, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        components.switch(frame, 3, 1, self.ui_state, "only_cache")

        # debug
        components.label(frame, 4, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        components.switch(frame, 4, 1, self.ui_state, "debug_mode")

        components.label(frame, 5, 0, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        components.dir_entry(frame, 5, 1, self.ui_state, "debug_dir")

        # tensorboard
        components.label(frame, 6, 0, "Tensorboard Mode",
                        tooltip="Off: Disabled.\nTrain Only: Active only during training.\nAlways On: Always available.")
        components.options_kv(frame, 6, 1, [
            ("Off", TensorboardMode.OFF),
            ("Train only", TensorboardMode.TRAIN_ONLY),
            ("Always on", TensorboardMode.ALWAYS_ON),
        ], self.ui_state, "tensorboard_mode")

        components.label(frame, 7, 0, "Tensorboard Port",
                         tooltip="Port to use for Tensorboard link")
        components.entry(frame, 7, 1, self.ui_state, "tensorboard_port")

        components.label(frame, 7, 2, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")
        components.switch(frame, 7, 3, self.ui_state, "tensorboard_expose")

        # validation
        components.label(frame, 9, 0, "Validation",
                         tooltip="Enable validation steps and add new graph in tensorboard")
        components.switch(frame, 9, 1, self.ui_state, "validation")

        components.label(frame, 10, 0, "Validate after",
                         tooltip="The interval used when validate training")
        components.time_entry(frame, 10, 1, self.ui_state, "validate_after", "validate_after_unit")

        # device
        components.label(frame, 11, 0, "Dataloader Threads",
                         tooltip="Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.")
        components.entry(frame, 11, 1, self.ui_state, "dataloader_threads")

        components.label(frame, 12, 0, "Train Device",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\"")
        components.entry(frame, 12, 1, self.ui_state, "train_device")

        components.label(frame, 13, 0, "Temp Device",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")
        components.entry(frame, 13, 1, self.ui_state, "temp_device")

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

    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)

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

        if self.train_config.cloud.enabled:
            trainer = CloudTrainer(self.train_config, self.training_callbacks, self.training_commands, reattach=self.cloud_tab.reattach)
        else:
            ZLUDA.initialize_devices(self.train_config)
            trainer = GenericTrainer(self.train_config, self.training_callbacks, self.training_commands)

        try:
            trainer.start()
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
            trainer.train()
        except Exception:
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
            error_caught = True
            traceback.print_exc()
        finally:
            trainer.end()

            # clear gpu memory
            del trainer

            self.training_thread = None
            self.training_commands = None
            torch.clear_autocast_cache()
            torch_gc()

        if error_caught:
            self.on_update_status("error: check the console for more information")
        else:
            self.on_update_status("stopped")

        self.training_button.configure(text="Start Training", state="normal")

    def start_training(self):
        if self.training_thread is None:
            self.save_default()

            self.training_button.configure(text="Stop Training", state="normal")

            self.training_commands = TrainCommands()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            self.training_button.configure(state="disabled")
            self.on_update_status("stopping")
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

    def _initialize_tensorboard(self):
        if self.train_config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
            self._start_always_on_tensorboard()

    def _start_always_on_tensorboard(self):
        if not self.always_on_tensorboard_subprocess:
            self.always_on_tensorboard_subprocess = start_filtered_tensorboard(self.train_config)

    def _stop_always_on_tensorboard(self):
        if self.always_on_tensorboard_subprocess:
            stop_tensorboard(self.always_on_tensorboard_subprocess)
            self.always_on_tensorboard_subprocess = None

    def _restart_always_on_tensorboard(self):
        """Restart always-on Tensorboard (for workspace directory changes)"""
        if self.train_config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
            self._stop_always_on_tensorboard()
            self._start_always_on_tensorboard()

    def _on_tensorboard_mode_change(self, *args):
        if self.train_config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
            self._start_always_on_tensorboard()
        else:
            self._stop_always_on_tensorboard()

    def _on_workspace_dir_change(self, new_workspace_dir: str):
        if new_workspace_dir != self.current_workspace_dir:
            self.current_workspace_dir = new_workspace_dir
            # Restart always-on Tensorboard with new workspace directory if it's enabled
            if self.train_config.tensorboard_mode == TensorboardMode.ALWAYS_ON:
                self._restart_always_on_tensorboard()

    def _on_workspace_dir_change_trace(self, *args):
        """Handle workspace directory changes via UI state trace"""
        new_workspace_dir = self.train_config.workspace_dir
        self._on_workspace_dir_change(new_workspace_dir)
