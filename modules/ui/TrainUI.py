import gc
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import filedialog
from typing import Callable

import customtkinter as ctk
import torch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.CaptionUI import CaptionUI
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.optimizer_util import UserPreferenceUtility, OPTIMIZER_KEY_MAP
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState


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

        self.train_args = TrainArgs.default_values()
        self.ui_state = UIState(self, self.train_args)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.status_label = None
        self.training_button = None
        self.tabview = None

        self.top_bar_component = self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

    def close(self):
        self.top_bar_component.save_default()

    def top_bar(self, master):
        return TopBar(master, self.train_args, self.ui_state, self.change_training_method)

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

        self.general_tab(self.tabview.add("general"))
        self.model_tab(self.tabview.add("model"))
        self.data_tab(self.tabview.add("data"))
        self.concepts_tab(self.tabview.add("concepts"))
        self.training_tab(self.tabview.add("training"))
        self.sampling_tab(self.tabview.add("sampling"))
        self.backup_tab(self.tabview.add("backup"))
        self.tools_tab(self.tabview.add("tools"))

        self.change_training_method(self.train_args.training_method)

        return frame

    def general_tab(self, master):
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

        components.label(master, 6, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        components.switch(master, 6, 1, self.ui_state, "tensorboard")

    def model_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # base model
        components.label(master, 0, 0, "Base Model",
                         tooltip="Filename, directory or hugging face repository of the base model")
        components.file_entry(
            master, 0, 1, self.ui_state, "base_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # model type
        components.label(master, 0, 3, "Model Type",
                         tooltip="Type of the base model")
        components.options_kv(master, 0, 4, [
            ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
            ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
            ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
            ("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
        ], self.ui_state, "model_type")

        # output model destination
        components.label(master, 1, 0, "Model Output Destination",
                         tooltip="Filename or directory where the output model is saved")
        components.file_entry(master, 1, 1, self.ui_state, "output_model_destination", is_output=True)

        # output format
        components.label(master, 1, 3, "Output Format",
                         tooltip="Format to use when saving the output model")
        components.options_kv(master, 1, 4, [
            ("Safetensors", ModelFormat.SAFETENSORS),
            ("Diffusers", ModelFormat.DIFFUSERS),
            ("Checkpoint", ModelFormat.CKPT),
        ], self.ui_state, "output_model_format")

        # output data type
        components.label(master, 2, 0, "Output Data Type",
                         tooltip="Precision to use when saving the output model")
        components.options_kv(master, 2, 1, [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "output_dtype")

        # weight dtype
        components.label(master, 3, 0, "Weight Data Type",
                         tooltip="The base model weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
        ], self.ui_state, "weight_dtype")

        # text encoder weight dtype
        components.label(master, 4, 0, "Override Text Encoder Data Type",
                         tooltip="Overrides the text encoder weight data type")
        components.options_kv(master, 4, 1, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
        ], self.ui_state, "text_encoder_weight_dtype")

        # unet weight dtype
        components.label(master, 5, 0, "Override UNet Data Type",
                         tooltip="Overrides the unet weight data type")
        components.options_kv(master, 5, 1, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
        ], self.ui_state, "unet_weight_dtype")

        # vae weight dtype
        components.label(master, 6, 0, "Override VAE Data Type",
                         tooltip="Overrides the vae weight data type")
        components.options_kv(master, 6, 1, [
            ("", DataType.NONE),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
        ], self.ui_state, "vae_weight_dtype")

    def data_tab(self, master):
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

        # latent caching epochs
        components.label(master, 4, 0, "Latent Caching Epochs",
                         tooltip="The number of epochs that are cached. Set this to a number higher than 1 and enable data augmentations if you want to add more diversity to your training data")
        components.entry(master, 4, 1, self.ui_state, "latent_caching_epochs")

        # clear cache before training
        components.label(master, 5, 0, "Clear cache before training",
                         tooltip="Clears the cache directory before starting to train. Only disable this if you want to continue using the same cached data. Disabling this can lead to errors, if other settings are changed during a restart")
        components.switch(master, 5, 1, self.ui_state, "clear_cache_before_training")

    def concepts_tab(self, master):
        ConceptTab(master, self.train_args, self.ui_state)

    def training_tab(self, master):
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        scroll_frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")

        scroll_frame.grid_columnconfigure(0, weight=0)
        scroll_frame.grid_columnconfigure(1, weight=1)
        scroll_frame.grid_columnconfigure(2, minsize=20)
        scroll_frame.grid_columnconfigure(3, weight=0)
        scroll_frame.grid_columnconfigure(4, weight=1)
        scroll_frame.grid_columnconfigure(5, minsize=20)
        scroll_frame.grid_columnconfigure(6, weight=0)
        scroll_frame.grid_columnconfigure(7, weight=1)

        # column 1
        # optimizer
        components.label(scroll_frame, 0, 0, "Optimizer",
                         tooltip="The type of optimizer")
        components.options_adv(scroll_frame, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer",
                               command=self.restore_optimizer_prefs, adv_command=self.open_optimizer_params_window)

        # learning rate scheduler
        components.label(scroll_frame, 1, 0, "Learning Rate Scheduler",
                         tooltip="Learning rate scheduler that automatically changes the learning rate during training")
        components.options(scroll_frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                           "learning_rate_scheduler")

        # learning rate
        components.label(scroll_frame, 2, 0, "Learning Rate",
                         tooltip="The base learning rate")
        components.entry(scroll_frame, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(scroll_frame, 3, 0, "Learning Rate Warmup Steps",
                         tooltip="The number of steps it takes to gradually increase the learning rate from 0 to the specified learning rate")
        components.entry(scroll_frame, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate cycles
        components.label(scroll_frame, 4, 0, "Learning Rate Cycles",
                         tooltip="The number of learning rate cycles. This is only applicable if the learning rate scheduler supports cycles")
        components.entry(scroll_frame, 4, 1, self.ui_state, "learning_rate_cycles")

        # epochs
        components.label(scroll_frame, 5, 0, "Epochs",
                         tooltip="The number of epochs for a full training run")
        components.entry(scroll_frame, 5, 1, self.ui_state, "epochs")

        # batch size
        components.label(scroll_frame, 6, 0, "Batch Size",
                         tooltip="The batch size of one training step")
        components.entry(scroll_frame, 6, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(scroll_frame, 7, 0, "Accumulation Steps",
                         tooltip="Number of accumulation steps. Increase this number to trade batch size for training speed")
        components.entry(scroll_frame, 7, 1, self.ui_state, "gradient_accumulation_steps")

        # attention mechanism
        components.label(scroll_frame, 8, 0, "Attention",
                         tooltip="The attention mechanism used during training. This has a big effect on speed and memory consumption")
        components.options(scroll_frame, 8, 1, [str(x) for x in list(AttentionMechanism)], self.ui_state,
                           "attention_mechanism")

        # ema
        components.label(scroll_frame, 9, 0, "EMA",
                         tooltip="EMA averages the training progress over many steps, better preserving different concepts in big datasets")
        components.options(scroll_frame, 9, 1, [str(x) for x in list(EMAMode)], self.ui_state,
                           "ema")

        # ema decay
        components.label(scroll_frame, 10, 0, "EMA Decay",
                         tooltip="Decay parameter of the EMA model. Higher numbers will average more steps. For datasets of hundreds or thousands of images, set this to 0.9999. For smaller datasets, set it to 0.999 or even 0.998")
        components.entry(scroll_frame, 10, 1, self.ui_state, "ema_decay")

        # ema update step interval
        components.label(scroll_frame, 11, 0, "EMA Update Step Interval",
                         tooltip="Number of steps between EMA update steps")
        components.entry(scroll_frame, 11, 1, self.ui_state, "ema_update_step_interval")

        # column 2
        # train text encoder
        components.label(scroll_frame, 0, 3, "Train Text Encoder",
                         tooltip="Enables training the text encoder model")
        components.switch(scroll_frame, 0, 4, self.ui_state, "train_text_encoder")

        # train text encoder epochs
        components.label(scroll_frame, 1, 3, "Train Text Encoder Epochs",
                         tooltip="Number of epochs to train the text encoder")
        components.entry(scroll_frame, 1, 4, self.ui_state, "train_text_encoder_epochs")

        # text encoder learning rate
        components.label(scroll_frame, 2, 3, "Text Encoder Learning Rate",
                         tooltip="The learning rate of the text encoder. Overrides the base learning rate")
        components.entry(scroll_frame, 2, 4, self.ui_state, "text_encoder_learning_rate")

        # text encoder layer skip (clip skip)
        components.label(scroll_frame, 3, 3, "Clip Skip",
                         tooltip="The number of clip layers to skip. 0 = disabled")
        components.entry(scroll_frame, 3, 4, self.ui_state, "text_encoder_layer_skip")

        # offset noise weight
        components.label(scroll_frame, 5, 3, "Offset Noise Weight",
                         tooltip="The weight of offset noise added to each training step")
        components.entry(scroll_frame, 5, 4, self.ui_state, "offset_noise_weight")

        # perturbation noise weight
        components.label(scroll_frame, 6, 3, "Perturbation Noise Weight",
                         tooltip="The weight of perturbation noise added to each training step")
        components.entry(scroll_frame, 6, 4, self.ui_state, "perturbation_noise_weight")

        # gradient checkpointing
        components.label(scroll_frame, 7, 3, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.switch(scroll_frame, 7, 4, self.ui_state, "gradient_checkpointing")

        # rescale noise scheduler to zero terminal SNR
        components.label(scroll_frame, 8, 3, "Rescale Noise Scheduler",
                         tooltip="Rescales the noise scheduler to a zero terminal signal to noise ratio and switches the model to a v-prediction target")
        components.switch(scroll_frame, 8, 4, self.ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")

        # train dtype
        components.label(scroll_frame, 9, 3, "Train Data Type",
                         tooltip="The mixed precision data type used for training. This can increase training speed, but reduces precision")
        components.options_kv(scroll_frame, 9, 4, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")

        # resolution
        components.label(scroll_frame, 10, 3, "Resolution",
                         tooltip="The resolution used for training")
        components.entry(scroll_frame, 10, 4, self.ui_state, "resolution")

        # column 3
        # train unet
        components.label(scroll_frame, 0, 6, "Train UNet",
                         tooltip="Enables training the U-Net model")
        components.switch(scroll_frame, 0, 7, self.ui_state, "train_unet")

        # train unet epochs
        components.label(scroll_frame, 1, 6, "Train UNet Epochs",
                         tooltip="Number of epochs to train the U-Net")
        components.entry(scroll_frame, 1, 7, self.ui_state, "train_unet_epochs")

        # unet learning rate
        components.label(scroll_frame, 2, 6, "Unet Learning Rate",
                         tooltip="The learning rate of the U-Net. Overrides the base learning rate")
        components.entry(scroll_frame, 2, 7, self.ui_state, "unet_learning_rate")

        # Masked Training
        components.label(scroll_frame, 5, 6, "Masked Training",
                         tooltip="Masks the training samples to let the model focus on certain parts of the image. When enabled, one mask image is loaded for each training sample.")
        components.switch(scroll_frame, 5, 7, self.ui_state, "masked_training")

        # unmasked probability
        components.label(scroll_frame, 6, 6, "Unmasked Probability",
                         tooltip="When masked training is enabled, specifies the number of training steps done on unmasked samples")
        components.entry(scroll_frame, 6, 7, self.ui_state, "unmasked_probability")

        # unmasked weight
        components.label(scroll_frame, 7, 6, "Unmasked Weight",
                         tooltip="When masked training is enabled, specifies the loss weight of areas outside the masked region")
        components.entry(scroll_frame, 7, 7, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(scroll_frame, 8, 6, "Normalize Masked Area Loss",
                         tooltip="When masked training is enabled, normalizes the loss for each sample based on the sizes of the masked region")
        components.switch(scroll_frame, 8, 7, self.ui_state, "normalize_masked_area_loss")

        # max noising strength
        components.label(scroll_frame, 9, 6, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(scroll_frame, 9, 7, self.ui_state, "max_noising_strength")

        # align prop
        components.label(scroll_frame, 11, 6, "AlignProp",
                         tooltip="Enables AlignProp training")
        components.switch(scroll_frame, 11, 7, self.ui_state, "align_prop")

        # align prop probability
        components.label(scroll_frame, 12, 6, "AlignProp Probability",
                         tooltip="When AlignProp is enabled, specifies the number of training steps done using AlignProp calculations")
        components.entry(scroll_frame, 12, 7, self.ui_state, "align_prop_probability")

        # align prop loss
        components.label(scroll_frame, 13, 6, "AlignProp Loss",
                         tooltip="Specifies the loss function used for AlignProp calculations")
        components.options(scroll_frame, 13, 7, [str(x) for x in list(AlignPropLoss)], self.ui_state, "align_prop_loss")

        # align prop weight
        components.label(scroll_frame, 14, 6, "AlignProp Weight",
                         tooltip="A weight multiplier for the AlignProp loss")
        components.entry(scroll_frame, 14, 7, self.ui_state, "align_prop_weight")

    def sampling_tab(self, master):
        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        # sample after
        top_frame = ctk.CTkFrame(master=master, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")

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

        # table
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        SamplingTab(frame, self.train_args, self.ui_state)

    def backup_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # backup after
        components.label(master, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        components.time_entry(master, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

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

    def lora_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # extra model
        components.label(master, 0, 0, "LoRA base model",
                         tooltip="The base LoRA to train on. Leave empty to create a new LoRA")
        components.file_entry(
            master, 0, 1, self.ui_state, "extra_model_name",
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

        # lora weight dtype
        components.label(master, 3, 0, "LoRA Weight Data Type",
                         tooltip="The LoRA weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 3, 1, [
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

        # extra model
        components.label(master, 0, 0, "Base embedding",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        components.file_entry(
            master, 0, 1, self.ui_state, "extra_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # token count
        components.label(master, 1, 0, "Token count",
                         tooltip="The token count used when creating a new embedding")
        components.entry(master, 1, 1, self.ui_state, "token_count")

        # initial embedding text
        components.label(master, 2, 0, "Initial embedding text",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(master, 2, 1, self.ui_state, "initial_embedding_text")

        # embedding weight dtype
        components.label(master, 3, 0, "Embedding Weight Data Type",
                         tooltip="The Embedding weight data type used for training. This can reduce memory consumption, but reduces precision")
        components.options_kv(master, 3, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "embedding_weight_dtype")

        return master

    def tools_tab(self, master):
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

        return master

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

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
        window = CaptionUI(self, None)
        self.wait_window(window)

    def open_convert_model_tool(self):
        window = ConvertModelUI(self)
        self.wait_window(window)

    def open_optimizer_params_window(self):
        window = OptimizerParamsWindow(self, self.ui_state)
        self.wait_window(window)

    def restore_optimizer_prefs(self, optimizer):
        pref_util = UserPreferenceUtility()
        user_prefs = pref_util.load_preferences(optimizer)

        for key, default_value in OPTIMIZER_KEY_MAP[optimizer].items():
            if user_prefs == "Use_Default":
                value_to_set = default_value
            else:
                value_to_set = user_prefs.get(key, default_value)

            self.ui_state.vars[key].set(value_to_set)

    def open_sample_ui(self):
        training_callbacks = self.training_callbacks
        training_commands = self.training_commands

        if training_callbacks and training_commands:
            window = SampleWindow(self, training_callbacks, training_commands)
            self.wait_window(window)
            training_callbacks.set_on_sample_custom()

    def __training_thread_function(self):
        error_caught = False

        self.training_callbacks = TrainCallbacks(
            on_update_train_progress=self.on_update_train_progress,
            on_update_status=self.on_update_status,
        )

        trainer = GenericTrainer(self.train_args, self.training_callbacks, self.training_commands)

        try:
            trainer.start()
            trainer.train()
        except:
            error_caught = True
            traceback.print_exc()

        trainer.end()

        # clear gpu memory
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

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
        args = self.train_args.to_args()
        command = "python scripts/train.py " + args

        file_path = filedialog.asksaveasfilename(filetypes=[
            ("All Files", "*.*"),
            ("Batch", "*.bat"),
            ("Shell", "*.sh"),
        ], initialdir=".", initialfile="train.bat")

        if file_path:
            with open(file_path, "w") as f:
                f.write(command)

    def sample_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.sample_default()
