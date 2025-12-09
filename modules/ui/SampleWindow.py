import contextlib
import copy
import os
import tkinter as tk
import traceback

from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import (
    BaseModelSampler,
    ModelSamplerOutput,
)
from modules.ui.SampleFrame import SampleFrame
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.time_util import get_string_timestamp
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import torch

import customtkinter as ctk
from PIL import Image


class SampleWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            train_config: TrainConfig | None = None,
            callbacks: TrainCallbacks | None = None,
            commands: TrainCommands | None = None,
            *args, **kwargs
    ):
        super().__init__(parent, *args, **kwargs)

        self.title("Sample")
        self.geometry("1200x800")
        self.resizable(True, True)

        if train_config is not None:
            self.initial_train_config = TrainConfig.default_values().from_dict(train_config.to_dict())
            # remove some settings to speed up model loading for sampling
            self.initial_train_config.optimizer.optimizer = None
            self.initial_train_config.ema = EMAMode.OFF
        else:
            self.initial_train_config = None

        self.current_train_config = train_config
        self.callbacks = callbacks
        self.commands = commands
        self.sample = SampleConfig.default_values()
        self.ui_state = UIState(self, self.sample)

        use_external_model = self.initial_train_config is None
        if use_external_model:
            self.callbacks.set_on_sample_custom(self.__update_preview)
            self.callbacks.set_on_update_sample_custom_progress(self.__update_progress)
        else:
            self.model = None
            self.model_sampler = None

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        prompt_frame = SampleFrame(self, self.sample, self.ui_state, include_settings=False)
        prompt_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="nsew")

        settings_frame = SampleFrame(self, self.sample, self.ui_state, include_prompt=False)
        settings_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

        # image
        self.image = ctk.CTkImage(
            light_image=self.__dummy_image(),
            size=(512, 512)
        )

        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=512, width=512)
        image_label.grid(row=1, column=1, rowspan=3, sticky="nsew")

        self.progress = components.progress(self, 2, 0)
        components.button(self, 3, 0, "sample", self.__sample)

        self.wait_visibility()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def __load_model(self) -> BaseModel:
        model_loader = create.create_model_loader(
            model_type=self.initial_train_config.model_type,
            training_method=self.initial_train_config.training_method,
        )

        model_setup = create.create_model_setup(
            model_type=self.initial_train_config.model_type,
            train_device=torch.device(self.initial_train_config.train_device),
            temp_device=torch.device(self.initial_train_config.temp_device),
            training_method=self.initial_train_config.training_method,
        )

        model_names = self.initial_train_config.model_names()
        if self.initial_train_config.continue_last_backup:
            last_backup_path = self.initial_train_config.get_last_backup_path()

            if last_backup_path:
                if self.initial_train_config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.initial_train_config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Loading from backup '{last_backup_path}'...")
            else:
                print("No backup found, loading without backup...")

        if self.initial_train_config.quantization.cache_dir is None:
            self.initial_train_config.quantization.cache_dir = self.initial_train_config.cache_dir + "/quantization"
            os.makedirs(self.initial_train_config.quantization.cache_dir, exist_ok=True)

        model = model_loader.load(
            model_type=self.initial_train_config.model_type,
            model_names=model_names,
            weight_dtypes=self.initial_train_config.weight_dtypes(),
            quantization=self.initial_train_config.quantization,
        )
        model.train_config = self.initial_train_config

        model_setup.setup_optimizations(model, self.initial_train_config)
        model_setup.setup_train_device(model, self.initial_train_config)
        model_setup.setup_model(model, self.initial_train_config)
        model.to(torch.device(self.initial_train_config.temp_device))

        return model

    def __create_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            train_device=torch.device(self.initial_train_config.train_device),
            temp_device=torch.device(self.initial_train_config.temp_device),
            model=model,
            model_type=self.initial_train_config.model_type,
            training_method=self.initial_train_config.training_method,
        )

    def __update_preview(self, sampler_output: ModelSamplerOutput):
        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            self.image.configure(
                light_image=image,
                size=(image.width, image.height),
            )

    def __update_progress(self, progress: int, max_progress: int):
        self.progress.set(progress / max_progress)
        self.update()

    def __dummy_image(self) -> Image:
        return Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

    def __sample(self):
        sample = copy.copy(self.sample)

        if self.commands:
            self.commands.sample_custom(sample)
        else:
            if self.model is None:
                # lazy initialization
                self.model = self.__load_model()
                self.model_sampler = self.__create_sampler(self.model)

            sample.from_train_config(self.current_train_config)

            sample_dir = os.path.join(
                self.initial_train_config.workspace_dir,
                "samples",
                "custom",
            )

            progress = self.model.train_progress
            sample_path = os.path.join(
                sample_dir,
                f"{get_string_timestamp()}-training-sample-{progress.filename_string()}"
            )

            self.model.eval()

            self.model_sampler.sample(
                sample_config=sample,
                destination=sample_path,
                image_format=self.current_train_config.sample_image_format,
                video_format=self.current_train_config.sample_video_format,
                audio_format=self.current_train_config.sample_audio_format,
                on_sample=self.__update_preview,
                on_update_progress=self.__update_progress,
            )

    def destroy(self):
        try:
            if hasattr(self, "_icon_image_ref"):
                del self._icon_image_ref

            # Remove any pending after callbacks
            for after_id in self.tk.call('after', 'info'):
                with contextlib.suppress(tk.TclError, RuntimeError):
                    self.after_cancel(after_id)

            super().destroy()
        except (tk.TclError, RuntimeError) as e:
            print(f"Error destroying window: {e}")
        except Exception as e:
            print(f"Unexpected error destroying window: {e}")
            traceback.print_exc()
