import copy
import os

import customtkinter as ctk
import torch
from PIL import Image

from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.ui.SampleFrame import SampleFrame
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.time_util import get_string_timestamp
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SampleWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            train_config: TrainConfig | None = None,
            callbacks: TrainCallbacks | None = None,
            commands: TrainCommands | None = None,
            *args, **kwargs
    ):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        if train_config is not None:
            self.train_config = TrainConfig.default_values().from_dict(train_config.to_dict())

            # remove some settings to speed up model loading for sampling
            self.train_config.optimizer.optimizer = None
            self.train_config.ema = EMAMode.OFF
        else:
            self.train_config = None
        self.callbacks = callbacks
        self.commands = commands

        use_external_model = self.train_config is None

        if use_external_model:
            self.callbacks.set_on_sample_custom(self.__update_preview)
            self.callbacks.set_on_update_sample_custom_progress(self.__update_progress)
        else:
            self.model = self.__load_model()
            self.model_sampler = self.__create_sampler(self.model)

        self.sample = SampleConfig.default_values()
        self.ui_state = UIState(self, self.sample)

        self.title("Sample")
        self.geometry("1200x800")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

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

    def __load_model(self) -> BaseModel:
        model_loader = create.create_model_loader(
            model_type=self.train_config.model_type,
            training_method=self.train_config.training_method,
        )

        model_setup = create.create_model_setup(
            model_type=self.train_config.model_type,
            train_device=torch.device(self.train_config.train_device),
            temp_device=torch.device(self.train_config.temp_device),
            training_method=self.train_config.training_method,
        )

        model = model_loader.load(
            model_type=self.train_config.model_type,
            model_names=self.train_config.model_names(),
            weight_dtypes=self.train_config.weight_dtypes(),
        )
        model.train_config = self.train_config

        model_setup.setup_model(model, self.train_config)

        return model

    def __create_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            train_device=torch.device(self.train_config.train_device),
            temp_device=torch.device(self.train_config.temp_device),
            model=model,
            model_type=self.train_config.model_type,
            training_method=self.train_config.training_method,
        )

    def __update_preview(self, image: Image):
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
        if self.commands:
            self.commands.sample_custom(copy.copy(self.sample))
        else:
            sample_dir = os.path.join(
                self.train_config.workspace_dir,
                "samples",
                "custom",
            )

            progress = self.model.train_progress
            image_format = self.train_config.sample_image_format
            sample_path = os.path.join(
                sample_dir,
                f"{get_string_timestamp()}-training-sample-{progress.filename_string()}{image_format.extension()}"
            )

            self.model_sampler.sample(
                sample_params=copy.copy(self.sample),
                destination=sample_path,
                image_format=self.train_config.sample_image_format,
                text_encoder_layer_skip=self.train_config.text_encoder_layer_skip,
                force_last_timestep=self.train_config.rescale_noise_scheduler_to_zero_terminal_snr,
                on_sample=self.__update_preview,
                on_update_progress=self.__update_progress,
            )
