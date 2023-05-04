import webbrowser
from typing import Callable

import customtkinter as ctk
import torch

from modules.trainer.FineTuneTrainer import FineTuneTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class TrainUI(ctk.CTk):
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]

    def __init__(self):
        super(TrainUI, self).__init__()

        self.train_args = TrainArgs.default_values()
        self.ui_state = UIState(self.train_args)

        self.title("OneTrainer")
        self.geometry("1000x700")

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # configure grid layout (4x4)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.top_bar(self)
        self.content_frame(self)
        self.bottom_bar(self)

    def top_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")

        # title
        components.app_title(frame, 0, 0)

        # padding
        frame.grid_columnconfigure(1, weight=1)

        # training method
        components.options_kv(
            frame, 0, 2, [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ], self.ui_state, "training_method"
        )

        return frame

    def bottom_bar(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=2, column=0, sticky="nsew")

        self.set_step_progress, self.set_epoch_progress = components.double_progress(frame, 0, 1, "step", "epoch")

        self.set_epoch_progress(1, 2)
        self.set_step_progress(10, 20)

        # padding
        frame.grid_columnconfigure(2, weight=1)

        # tensorboard button
        components.button(frame, 0, 3, "Tensorboard", self.open_tensorboard)

        # training button
        components.button(frame, 0, 4, "Start Training", self.start_training)

        return frame

    def content_frame(self, master):
        frame = ctk.CTkFrame(master=master, corner_radius=0)
        frame.grid(row=1, column=0, sticky="nsew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(frame)
        tabview.grid(row=0, column=0, sticky="nsew")

        self.general_tab(tabview.add("general"))
        self.model_tab(tabview.add("model"))
        self.data_tab(tabview.add("data"))
        self.training_tab(tabview.add("training"))
        self.sampling_tab(tabview.add("sampling"))
        self.backup_tab(tabview.add("backup"))
        self.tools_tab(tabview.add("tools"))

        return frame

    def general_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=0)
        master.grid_columnconfigure(3, weight=1)

        # workspace dir
        components.label(master, 0, 0, "Workspace Directory")
        components.file_entry(master, 0, 1, self.ui_state, "workspace_dir")

        # cache dir
        components.label(master, 1, 0, "Cache Directory")
        components.file_entry(master, 1, 1, self.ui_state, "cache_dir")

        # debug
        components.label(master, 2, 0, "Debug mode")
        components.switch(master, 2, 1, self.ui_state, "debug_mode")

        components.label(master, 3, 0, "Debug Directory")
        components.file_entry(master, 3, 1, self.ui_state, "debug_dir")

        components.label(master, 4, 0, "Tensorboard")
        components.switch(master, 4, 1, self.ui_state, "tensorboard")

    def model_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # base model
        components.label(master, 0, 0, "Base Model")
        components.file_entry(master, 0, 1, self.ui_state, "base_model_name")

        # model type
        components.label(master, 0, 3, "Model Type")
        components.options(master, 0, 4, [str(x) for x in list(ModelType)], self.ui_state, "model_type")

        # extra model
        components.label(master, 1, 0, "Extra Model")
        components.file_entry(master, 1, 1, self.ui_state, "extra_model_name")

    def data_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # base model
        components.label(master, 0, 0, "Concept Definition")
        components.file_entry(master, 0, 1, self.ui_state, "concept_file_name")

        # output format
        components.label(master, 1, 0, "Output Format")
        components.options(master, 1, 1, [str(x) for x in list(ModelFormat)], self.ui_state, "output_model_format")

        # extra model
        components.label(master, 2, 0, "Model Output Destination")
        components.file_entry(master, 2, 1, self.ui_state, "output_model_destination")

        # circular mask generation
        components.label(master, 3, 0, "Circular Mask Generation")
        components.switch(master, 3, 1, self.ui_state, "circular_mask_generation")

        # random rotate and crop
        components.label(master, 4, 0, "Random Rotate and Crop")
        components.switch(master, 4, 1, self.ui_state, "random_rotate_and_crop")

        # aspect ratio bucketing
        components.label(master, 5, 0, "Aspect Ratio Bucketing")
        components.switch(master, 5, 1, self.ui_state, "aspect_ratio_bucketing")

        # latent caching
        components.label(master, 6, 0, "Latent Caching")
        components.switch(master, 6, 1, self.ui_state, "latent_caching")

        # latent caching epochs
        components.label(master, 7, 0, "Latent Caching Epochs")
        components.entry(master, 7, 1, self.ui_state, "latent_caching_epochs")

    def training_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)
        master.grid_columnconfigure(5, minsize=50)
        master.grid_columnconfigure(6, weight=0)
        master.grid_columnconfigure(7, weight=1)

        # column 1
        # optimizer
        components.label(master, 0, 0, "Optimizer")
        components.options(master, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer")

        # learning rate
        components.label(master, 1, 0, "Learning Rate")
        components.entry(master, 1, 1, self.ui_state, "learning_rate")

        # weight decay
        components.label(master, 2, 0, "Weight Decay")
        components.entry(master, 2, 1, self.ui_state, "weight_decay")

        # loss function
        components.label(master, 3, 0, "Loss Function")
        components.options(master, 3, 1, [str(x) for x in list(LossFunction)], self.ui_state, "loss_function")

        # epochs
        components.label(master, 4, 0, "Epochs")
        components.entry(master, 4, 1, self.ui_state, "epochs")

        # batch size
        components.label(master, 5, 0, "Batch Size")
        components.entry(master, 5, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(master, 6, 0, "Accumulation Steps")
        components.entry(master, 6, 1, self.ui_state, "gradient_accumulation_steps")

        # column 2
        # train text encoder
        components.label(master, 0, 3, "Train Text Encoder")
        components.switch(master, 0, 4, self.ui_state, "train_text_encoder")

        # train text encoder epochs
        components.label(master, 1, 3, "Train Text Encoder Epochs")
        components.entry(master, 1, 4, self.ui_state, "train_text_encoder_epochs")

        # text encoder learning rate
        components.label(master, 2, 3, "Text Encoder Learning Rate")
        components.entry(master, 2, 4, self.ui_state, "text_encoder_learning_rate")

        # offset noise weight
        components.label(master, 4, 3, "Offset Noise Weight")
        components.entry(master, 4, 4, self.ui_state, "offset_noise_weight")

        # train dtype
        components.label(master, 5, 3, "Train Data Type")
        components.options_kv(master, 5, 4, [
            ("float16", torch.float16),
            ("float32", torch.float32)
        ], self.ui_state, "train_dtype")

        # resolution
        components.label(master, 6, 3, "Resolution")
        components.entry(master, 6, 4, self.ui_state, "resolution")

        # column 3
        # train unet
        components.label(master, 0, 6, "Train UNet")
        components.switch(master, 0, 7, self.ui_state, "train_unet")

        # train unet epochs
        components.label(master, 1, 6, "Train UNet Epochs")
        components.entry(master, 1, 7, self.ui_state, "train_unet_epochs")

        # unet learning rate
        components.label(master, 2, 6, "Unet Learning Rate")
        components.entry(master, 2, 7, self.ui_state, "unet_learning_rate")

        # Masked Training
        components.label(master, 4, 6, "Masked Training")
        components.switch(master, 4, 7, self.ui_state, "masked_training")

        # unmasked weight
        components.label(master, 5, 6, "Unmasked Weight")
        components.entry(master, 5, 7, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(master, 6, 6, "Normalize Masked Area Loss")
        components.entry(master, 6, 7, self.ui_state, "normalize_masked_area_loss")

        # max noising strength
        components.label(master, 7, 6, "Max Noising Strength")
        components.entry(master, 7, 7, self.ui_state, "max_noising_strength")

    def sampling_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # optimizer
        components.label(master, 0, 0, "Sample Definition")
        components.file_entry(master, 0, 1, self.ui_state, "sample_definition_file_name")

        # sample after
        components.label(master, 1, 0, "Sample After")
        components.time_entry(master, 1, 1, self.ui_state, "sample_after", "sample_after_unit")

    def backup_tab(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50)
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        # sample after
        components.label(master, 0, 0, "Backup After")
        components.time_entry(master, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        # optimizer
        components.label(master, 1, 0, "Backup Before Save")
        components.switch(master, 1, 1, self.ui_state, "backup_before_save")

    def tools_tab(self, master):
        pass

    def open_tensorboard(self):
        webbrowser.open("http://localhost:6006/", new=0, autoraise=False)

    def start_training(self):
        callbacks = TrainCallbacks()

        trainer = None
        match self.train_args.training_method:
            case TrainingMethod.FINE_TUNE:
                trainer = FineTuneTrainer(self.train_args, callbacks)
            case TrainingMethod.LORA:
                trainer = FineTuneTrainer(self.train_args, callbacks)
            case TrainingMethod.EMBEDDING:
                trainer = FineTuneTrainer(self.train_args, callbacks)
            case TrainingMethod.FINE_TUNE_VAE:
                trainer = FineTuneTrainer(self.train_args, callbacks)

        trainer.start()
        trainer.train()
        trainer.end()
