import copy
import os

from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import (
    BaseModelSampler,
)
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.time_util import get_string_timestamp

import torch


class SampleWindowController:
    def __init__(
            self,
            train_config: TrainConfig,
            use_external_model: bool,
            callbacks: TrainCallbacks | None = None,
            commands: TrainCommands | None = None,
    ):
        self.current_train_config = train_config
        self.use_external_model = use_external_model
        self.callbacks = callbacks
        self.commands = commands

        if not use_external_model:
            self.initial_train_config = TrainConfig.default_values().from_dict(train_config.to_dict())
            # remove some settings to speed up model loading for sampling
            self.initial_train_config.optimizer.optimizer = None
            self.initial_train_config.ema = EMAMode.OFF
        else:
            self.initial_train_config = None

        #TODO why is there a current_train_config and an initial_train_config?
        #current_train_config doesn't seem to ever change

        # get model specific defaults
        model_type = train_config.model_type
        self.sample = SampleConfig.default_values(model_type)

        if not use_external_model:
            self.model = None
            self.model_sampler = None

    def get_model_type(self):
        return self.current_train_config.model_type

    def load_model(self) -> BaseModel:
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

    def create_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(
            train_device=torch.device(self.initial_train_config.train_device),
            temp_device=torch.device(self.initial_train_config.temp_device),
            model=model,
            model_type=self.initial_train_config.model_type,
            training_method=self.initial_train_config.training_method,
        )

    def do_sample(self, on_sample, on_update_progress):
        sample = copy.copy(self.sample)

        if self.commands:
            self.commands.sample_custom(sample)
        else:
            if self.model is None:
                # lazy initialization
                self.model = self.load_model()
                self.model_sampler = self.create_sampler(self.model)

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
                on_sample=on_sample,
                on_update_progress=on_update_progress,
            )
