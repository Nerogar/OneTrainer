import json
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk

from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.CaptionUI import CaptionUI
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.ModelTab import ModelTab
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.ui.TrainingTab import TrainingTab
from modules.util.TrainProgress import TrainProgress
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.ui import components
from modules.util.ui.UIState import UIState
from modules.zluda import ZLUDA


class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    status_label: ctk.CTkLabel | None
    training_button: ctk.CTkButton | None
    training_callbacks: TrainCallbacks | None
    training_commands: TrainCommands | None

    def __init__(self):
        super(TrainUI, self).__init__()

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

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

    def close(self):
        self.top_bar_component.save_default()

    def top_bar(self, master):
        return TopBar(master, self.train_config, self.ui_state, self.change_model_type, self.change_training_method)

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

        self.create_general_tab(self.tabview.add("general"))
        self.model_tab = self.create_model_tab(self.tabview.add("model"))
        self.create_data_tab(self.tabview.add("data"))
        self.create_concepts_tab(self.tabview.add("concepts"))
        self.training_tab = self.create_training_tab(self.tabview.add("training"))
        self.create_sampling_tab(self.tabview.add("sampling"))
        self.create_backup_tab(self.tabview.add("backup"))
        self.create_tools_tab(self.tabview.add("tools"))

        self.change_training_method(self.train_config.training_method)

        return frame

    def create_general_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=0)
        master.grid_columnconfigure(3, weight=1)

        # workspace dir
        components.label(master, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run are saved")
        components.dir_entry(master, 0, 1, self.ui_state, "workspace_dir")

        # cache dir
        components.label(master, 1, 0, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        components.dir_entry(master, 1, 1, self.ui_state, "cache_dir")

        # continue from previous backup
        components.label(master, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/run/backup")
        components.switch(master, 2, 1, self.ui_state, "continue_last_backup")

        # only cache
        components.label(master, 3, 0, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        components.switch(master, 3, 1, self.ui_state, "only_cache")

        # debug
        components.label(master, 4, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        components.switch(master, 4, 1, self.ui_state, "debug_mode")

        components.label(master, 5, 0, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        components.dir_entry(master, 5, 1, self.ui_state, "debug_dir")

        # tensorboard
        components.label(master, 6, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        components.switch(master, 6, 1, self.ui_state, "tensorboard")

        components.label(master, 7, 0, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")
        components.switch(master, 7, 1, self.ui_state, "tensorboard_expose")

        # device
        components.label(master, 8, 0, "Train Device",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\"")
        components.entry(master, 8, 1, self.ui_state, "train_device")

        components.label(master, 9, 0, "Temp Device",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")
        components.entry(master, 9, 1, self.ui_state, "temp_device")

    def create_model_tab(self, master):
        return ModelTab(master, self.train_config, self.ui_state)

    def create_data_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # circular mask generation
        components.label(master, 0, 0, "Circular Mask Generation",
                         tooltip="Automatically create circular masks for masked training")
        components.switch(master, 0, 1, self.ui_state, "circular_mask_generation")

        # random rotate and crop
        components.label(master, 1, 0, "Random Rotate and Crop",
                         tooltip="Randomly rotate the training samples and crop to the masked region")
        components.switch(master, 1, 1, self.ui_state, "random_rotate_and_crop")

        # aspect ratio bucketing
        components.label(master, 2, 0, "Aspect Ratio Bucketing",
                         tooltip="Aspect ratio bucketing enables training on images with different aspect ratios")
        components.switch(master, 2, 1, self.ui_state, "aspect_ratio_bucketing")

        # latent caching
        components.label(master, 3, 0, "Latent Caching",
                         tooltip="Caching of intermediate training data that can be re-used between epochs")
        components.switch(master, 3, 1, self.ui_state, "latent_caching")

        # clear cache before training
        components.label(master, 4, 0, "Clear cache before training",
                         tooltip="Clears the cache directory before starting to train. Only disable this if you want to continue using the same cached data. Disabling this can lead to errors, if other settings are changed during a restart")
        components.switch(master, 4, 1, self.ui_state, "clear_cache_before_training")

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
        components.label(top_frame, 0, 0, "Sample After",
                         tooltip="The interval used when automatically sampling from the model during training")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")

        components.label(top_frame, 0, 2, "Format",
                         tooltip="File Format used when saving samples")
        components.options_kv(top_frame, 0, 3, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], self.ui_state, "sample_image_format")

        components.button(top_frame, 0, 4, "sample now", self.sample_now)

        components.button(top_frame, 0, 5, "manual sample", self.open_sample_ui)

        components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                         tooltip="Whether to include non-ema sampling when using ema.")
        components.switch(sub_frame, 0, 1, self.ui_state, "non_ema_sampling")

        components.label(sub_frame, 0, 2, "Samples to Tensorboard",
                         tooltip="Whether to include sample images in the Tensorboard output.")
        components.switch(sub_frame, 0, 3, self.ui_state, "samples_to_tensorboard")

        # table
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        SamplingTab(frame, self.train_config, self.ui_state)

    def create_backup_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # backup after
        components.label(master, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        components.time_entry(master, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        # backup now
        components.button(master, 0, 3, "backup now", self.backup_now)

        # rolling backup
        components.label(master, 1, 0, "Rolling Backup",
                         tooltip="If rolling backups are enabled, older backups are deleted automatically")
        components.switch(master, 1, 1, self.ui_state, "rolling_backup")

        # rolling backup count
        components.label(master, 1, 3, "Rolling Backup Count",
                         tooltip="Defines the number of backups to keep if rolling backups are enabled")
        components.entry(master, 1, 4, self.ui_state, "rolling_backup_count")

        # backup before save
        components.label(master, 2, 0, "Backup Before Save",
                         tooltip="Create a full backup before saving the final model")
        components.switch(master, 2, 1, self.ui_state, "backup_before_save")

        # save after
        components.label(master, 3, 0, "Save After",
                         tooltip="The interval used when automatically saving the model during training")
        components.time_entry(master, 3, 1, self.ui_state, "save_after", "save_after_unit")

        # save filename prefix
        components.label(master, 3, 3, "Save Filename Prefix",
                         tooltip="The prefix for filenames used when saving the model during training")
        components.entry(master, 3, 4, self.ui_state, "save_filename_prefix")


    def lora_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # lora model name
        components.label(master, 0, 0, "LoRA base model",
                         tooltip="The base LoRA to train on. Leave empty to create a new LoRA")
        components.file_entry(
            master, 0, 1, self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # lora rank
        components.label(master, 1, 0, "LoRA rank",
                         tooltip="The rank parameter used when creating a new LoRA")
        components.entry(master, 1, 1, self.ui_state, "lora_rank")

        # lora rank
        components.label(master, 2, 0, "LoRA alpha",
                         tooltip="The alpha parameter used when creating a new LoRA")
        components.entry(master, 2, 1, self.ui_state, "lora_alpha")

        # Dropout Percentage
        components.label(master, 3, 0, "Dropout Probability",
                         tooltip="Dropout probability. This percentage of model nodes will be randomly ignored at each training step. Helps with overfitting. 0 disables, 1 maximum.")
        components.entry(master, 3, 1, self.ui_state, "dropout_probability")

        # lora weight dtype
        components.label(master, 5, 0, "LoRA Weight Data Type",
                         tooltip="The LoRA weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 5, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "lora_weight_dtype")

        return master

    def embedding_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # embedding model names
        components.label(master, 0, 0, "Base embedding",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        components.file_entry(
            master, 0, 1, self.ui_state, "embeddings.model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # token count
        components.label(master, 1, 0, "Token count",
                         tooltip="The token count used when creating a new embedding")
        components.entry(master, 1, 1, self.ui_state, "embeddings.token_count")

        # initial embedding text
        components.label(master, 2, 0, "Initial embedding text",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(master, 2, 1, self.ui_state, "embeddings.initial_embedding_text")

        # embedding weight dtype
        components.label(master, 3, 0, "Embedding Weight Data Type",
                         tooltip="The Embedding weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "embedding_weight_dtype")

        return master

    def create_tools_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # dataset
        components.label(master, 0, 0, "Dataset Tools",
                         tooltip="Open the captioning tool")
        components.button(master, 0, 1, "Open", self.open_dataset_tool)

        # convert model
        components.label(master, 1, 0, "Convert Model Tools",
                         tooltip="Open the model conversion tool")
        components.button(master, 1, 1, "Open", self.open_convert_model_tool)

        # sample
        components.label(master, 2, 0, "Sampling Tool",
                         tooltip="Open the model sampling tool")
        components.button(master, 2, 1, "Open", self.open_sampling_tool)

        return master

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()

        if self.training_tab:
            self.training_tab.refresh_ui()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        if training_method != TrainingMethod.LORA and "LoRA" in self.tabview._tab_dict:
            self.tabview.delete("LoRA")
        if training_method != TrainingMethod.EMBEDDING and "embedding" in self.tabview._tab_dict:
            self.tabview.delete("embedding")

        if training_method == TrainingMethod.LORA and "LoRA" not in self.tabview._tab_dict:
            self.lora_tab(self.tabview.add("LoRA"))
        if training_method == TrainingMethod.EMBEDDING and "embedding" not in self.tabview._tab_dict:
            self.embedding_tab(self.tabview.add("embedding"))

    def open_tensorboard(self):
        webbrowser.open("http://localhost:6006/", new=0, autoraise=False)

    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)
        pass

    def on_update_status(self, status: str):
        self.status_label.configure(text=status)
        pass

    def open_dataset_tool(self):
        window = CaptionUI(self, None, False)
        self.wait_window(window)

    def open_convert_model_tool(self):
        window = ConvertModelUI(self)
        self.wait_window(window)

    def open_sampling_tool(self):
        window = SampleWindow(
            self,
            train_config=self.train_config,
        )
        self.wait_window(window)
        torch_gc()

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
        except:
            error_caught = True
            traceback.print_exc()

        trainer.end()

        # clear gpu memory
        torch_gc()

        if error_caught:
            self.on_update_status("error: check the console for more information")
        else:
            self.on_update_status("stopped")

        self.training_thread = None
        self.training_commands = None
        self.training_button.configure(text="Start Training", state="normal")

    def start_training(self):
        if self.training_thread is None:
            self.top_bar_component.save_default()

            self.training_button.configure(text="Stop Training", state="normal")

            self.training_commands = TrainCommands()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            self.training_button.configure(state="disabled")
            self.on_update_status("stopping")
            self.training_commands.stop()

    def export_training(self):
        file_path = filedialog.asksaveasfilename(filetypes=[
            ("All Files", "*.*"),
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
