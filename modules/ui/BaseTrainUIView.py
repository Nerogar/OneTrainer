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
        self.ui_state.get_var("secrets.cloud").update(self.controller.train_config.secrets.cloud)

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
                                             tooltip="Current status of the training run")
        self.eta_label = self.components.label(status_frame, 1, 0, "", pad=0)

        self.export_button = self.components.button(frame, 0, 3, "Export", self.export_training,
                                             width=60, padx=5, pady=(15, 0),
                                             tooltip="Export the current configuration as a script to run without a UI")

        self.components.button(frame, 0, 4, "Debug", self.generate_debug_package,
                                           width=60, padx=(5, 25), pady=(15, 0),
                                           tooltip="Generate a zip file with config.json, debug_report.log and settings diff, use this to report bugs or issues")

        self.components.button(frame, 0, 5, "Tensorboard", self.open_tensorboard,
                                           width=100, padx=(0, 5), pady=(15, 0))

        self.training_button = self.components.button(frame, 0, 6, "Start Training", self.start_training,
                                                 padx=(5, 20), pady=(15, 0))

    def build_general_tab_content(self, frame, controller, ui_state):
        # workspace dir
        self.components.label(frame, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run are saved")
        self.components.path_entry(frame, 0, 1, ui_state, "workspace_dir", mode="dir", command=controller._on_workspace_dir_change)

        # cache dir
        self.components.label(frame, 0, 2, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        self.components.path_entry(frame, 0, 3, ui_state, "cache_dir", mode="dir")

        # continue from previous backup
        self.components.label(frame, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/backup")
        self.components.switch(frame, 2, 1, ui_state, "continue_last_backup")

        # only cache
        self.components.label(frame, 2, 2, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        self.components.switch(frame, 2, 3, ui_state, "only_cache")

        # TODO: In Phase 4 rework the general tab.
        # prevent overwrites
        self.components.label(frame, 3, 0, "Prevent Overwrites",
                         tooltip="When enabled, output paths that already exist on disk will be flagged as invalid to avoid accidental overwrites")
        self.components.switch(frame, 3, 1, ui_state, "prevent_overwrites")

        # debug
        self.components.label(frame, 4, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        self.components.switch(frame, 4, 1, ui_state, "debug_mode")

        self.components.label(frame, 4, 2, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        self.components.path_entry(frame, 4, 3, ui_state, "debug_dir", mode="dir", io_type=PathIOType.OUTPUT)

        # tensorboard
        self.components.label(frame, 6, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        self.components.switch(frame, 6, 1, ui_state, "tensorboard")

        self.components.label(frame, 6, 2, "Always-On Tensorboard",
                         tooltip="Keep Tensorboard accessible even when not training. Useful for monitoring completed training sessions.")
        self.components.switch(frame, 6, 3, ui_state, "tensorboard_always_on", command=controller._on_always_on_tensorboard_toggle)

        self.components.label(frame, 7, 0, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")
        self.components.switch(frame, 7, 1, ui_state, "tensorboard_expose")
        self.components.label(frame, 7, 2, "Tensorboard Port",
                         tooltip="Port to use for Tensorboard link")
        self.components.entry(frame, 7, 3, ui_state, "tensorboard_port")

        # validation
        self.components.label(frame, 8, 0, "Validation",
                         tooltip="Enable validation steps and add new graph in tensorboard")
        self.components.switch(frame, 8, 1, ui_state, "validation")

        self.components.label(frame, 8, 2, "Validate after",
                         tooltip="The interval used when validate training")
        self.components.time_entry(frame, 8, 3, ui_state, "validate_after", "validate_after_unit")

        # device
        self.components.label(frame, 10, 0, "Dataloader Threads",
                         tooltip="Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.")
        self.components.entry(frame, 10, 1, ui_state, "dataloader_threads", required=True)

        self.components.label(frame, 11, 0, "Train Device",
                         tooltip="The device used for training. Can be \"cuda\", \"cuda:0\", \"cuda:1\" etc. Default:\"cuda\". Must be \"cuda\" for multi-GPU training.")
        self.components.entry(frame, 11, 1, ui_state, "train_device", required=True)

        self.components.label(frame, 12, 0, "Multi-GPU",
                         tooltip="Enable multi-GPU training")
        self.components.switch(frame, 12, 1, ui_state, "multi_gpu")
        self.components.label(frame, 12, 2, "Device Indexes",
                         tooltip="Multi-GPU: A comma-separated list of device indexes. If empty, all your GPUs are used. With a list such as \"0,1,3,4\" you can omit a GPU, for example an on-board graphics GPU.")
        self.components.entry(frame, 12, 3, ui_state, "device_indexes")

        self.components.label(frame, 13, 0, "Gradient Reduce Precision",
                         tooltip="WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
                                 "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
                                 "FLOAT_32: Reduce gradients in float32\n"
                                 "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
                         wide_tooltip=True)
        self.components.options(frame, 13, 1, [str(x) for x in list(GradientReducePrecision)], ui_state,
                           "gradient_reduce_precision")

        self.components.label(frame, 13, 2, "Fused Gradient Reduce",
                         tooltip="Multi-GPU: Gradient synchronisation during the backward pass. Can be more efficient, especially with Async Gradient Reduce")
        self.components.switch(frame, 13, 3, ui_state, "fused_gradient_reduce")

        self.components.label(frame, 14, 0, "Async Gradient Reduce",
                         tooltip="Multi-GPU: Asynchroniously start the gradient reduce operations during the backward pass. Can be more efficient, but requires some VRAM.")
        self.components.switch(frame, 14, 1, ui_state, "async_gradient_reduce")
        self.components.label(frame, 14, 2, "Buffer size (MB)",
                         tooltip="Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"")
        self.components.entry(frame, 14, 3, ui_state, "async_gradient_reduce_buffer")

        self.components.label(frame, 15, 0, "Temp Device",
                         tooltip="The device used to temporarily offload models while they are not used. Default:\"cpu\"")
        self.components.entry(frame, 15, 1, ui_state, "temp_device")

    def build_data_tab_content(self, frame, controller, ui_state):
        # aspect ratio bucketing
        self.components.label(frame, 0, 0, "Aspect Ratio Bucketing",
                         tooltip="Aspect ratio bucketing enables training on images with different aspect ratios")
        self.components.switch(frame, 0, 1, ui_state, "aspect_ratio_bucketing")

        # latent caching
        self.components.label(frame, 1, 0, "Latent Caching",
                         tooltip="Caching of intermediate training data that can be re-used between epochs")
        self.components.switch(frame, 1, 1, ui_state, "latent_caching")

        # clear cache before training
        self.components.label(frame, 2, 0, "Clear cache before training",
                         tooltip="Clears the cache directory before starting to train. Only disable this if you want to continue using the same cached data. Disabling this can lead to errors, if other settings are changed during a restart")
        self.components.switch(frame, 2, 1, ui_state, "clear_cache_before_training")

    def build_sampling_tab_header(self, top_frame, sub_frame, controller, ui_state):
        self.components.label(top_frame, 0, 0, "Sample After",
                         tooltip="The interval used when automatically sampling from the model during training")
        self.components.time_entry(top_frame, 0, 1, ui_state, "sample_after", "sample_after_unit")

        self.components.label(top_frame, 0, 2, "Skip First",
                         tooltip="Start sampling automatically after this interval has elapsed.")
        self.components.entry(top_frame, 0, 3, ui_state, "sample_skip_first", width=50, sticky="nw")

        self.components.label(top_frame, 0, 4, "Format",
                         tooltip="File Format used when saving samples")
        self.components.options_kv(top_frame, 0, 5, [
            ("PNG", ImageFormat.PNG),
            ("JPG", ImageFormat.JPG),
        ], ui_state, "sample_image_format")

        self.components.button(top_frame, 0, 6, "sample now", self.sample_now)

        self.components.button(top_frame, 0, 7, "manual sample", self.open_manual_sample_window)

        self.components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                         tooltip="Whether to include non-ema sampling when using ema.")
        self.components.switch(sub_frame, 0, 1, ui_state, "non_ema_sampling")

        self.components.label(sub_frame, 0, 2, "Samples to Tensorboard",
                         tooltip="Whether to include sample images in the Tensorboard output.")
        self.components.switch(sub_frame, 0, 3, ui_state, "samples_to_tensorboard")

    def build_backup_tab_content(self, frame, controller, ui_state):
        # backup after
        self.components.label(frame, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        self.components.time_entry(frame, 0, 1, ui_state, "backup_after", "backup_after_unit")

        # backup now
        self.components.button(frame, 0, 3, "backup now", self.backup_now)

        # rolling backup
        self.components.label(frame, 1, 0, "Rolling Backup",
                         tooltip="If rolling backups are enabled, older backups are deleted automatically")
        self.components.switch(frame, 1, 1, ui_state, "rolling_backup")

        # rolling backup count
        self.components.label(frame, 2, 0, "Rolling Backup Count",
                         tooltip="Defines the number of backups to keep if rolling backups are enabled")
        self.components.entry(frame, 2, 1, ui_state, "rolling_backup_count")

        # backup before save
        self.components.label(frame, 3, 0, "Backup Before Save",
                         tooltip="Create a full backup before saving the final model")
        self.components.switch(frame, 3, 1, ui_state, "backup_before_save")

        # save after
        self.components.label(frame, 4, 0, "Save Every",
                         tooltip="The interval used when automatically saving the model during training")
        self.components.time_entry(frame, 4, 1, ui_state, "save_every", "save_every_unit")

        # save now
        self.components.button(frame, 4, 3, "save now", self.save_now)

        # skip save
        self.components.label(frame, 5, 0, "Skip First",
                         tooltip="Start saving automatically after this interval has elapsed")
        self.components.entry(frame, 5, 1, ui_state, "save_skip_first", width=50, sticky="nw")

        # save filename prefix
        self.components.label(frame, 6, 0, "Save Filename Prefix",
                         tooltip="The prefix for filenames used when saving the model during training")
        self.components.entry(frame, 6, 1, ui_state, "save_filename_prefix")

    def build_embedding_tab_content(self, frame, controller, ui_state):
        # embedding model name
        self.components.label(frame, 0, 0, "Base embedding",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        self.components.path_entry(
            frame, 0, 1, ui_state, "embedding.model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # token count
        self.components.label(frame, 1, 0, "Token count",
                         tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        self.components.entry(frame, 1, 1, ui_state, "embedding.token_count")

        # initial embedding text
        self.components.label(frame, 2, 0, "Initial embedding text",
                         tooltip="The initial embedding text used when creating a new embedding")
        self.components.entry(frame, 2, 1, ui_state, "embedding.initial_embedding_text")

        # embedding weight dtype
        self.components.label(frame, 3, 0, "Embedding Weight Data Type",
                         tooltip="The Embedding weight data type used for training. This can reduce memory consumption, but reduces precision")
        self.components.options_kv(frame, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], ui_state, "embedding_weight_dtype")

        # placeholder
        self.components.label(frame, 4, 0, "Placeholder",
                         tooltip="The placeholder used when using the embedding in a prompt")
        self.components.entry(frame, 4, 1, ui_state, "embedding.placeholder")

        # output embedding
        self.components.label(frame, 5, 0, "Output embedding",
                         tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        self.components.switch(frame, 5, 1, ui_state, "embedding.is_output_embedding")

    def build_tools_tab_content(self, frame, controller, ui_state):
        # dataset
        self.components.label(frame, 0, 0, "Dataset Tools",
                         tooltip="Open the captioning tool")
        self.components.button(frame, 0, 1, "Open", self.open_dataset_tool)

        # video tools
        self.components.label(frame, 1, 0, "Video Tools",
                         tooltip="Open the video tools")
        self.components.button(frame, 1, 1, "Open", self.open_video_tool)

        # convert model
        self.components.label(frame, 2, 0, "Convert Model Tools",
                         tooltip="Open the model conversion tool")
        self.components.button(frame, 2, 1, "Open", self.open_convert_model_tool)

        # sample
        self.components.label(frame, 3, 0, "Sampling Tool",
                         tooltip="Open the model sampling tool")
        self.components.button(frame, 3, 1, "Open", self.open_sampling_tool)

        self.components.label(frame, 4, 0, "Profiling Tool",
                         tooltip="Open the profiling tools.")
        self.components.button(frame, 4, 1, "Open", self.open_profiling_tool)
