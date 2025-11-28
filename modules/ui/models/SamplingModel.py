import copy
import os

from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import ModelSamplerOutput
from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.ui.models.StateModel import StateModel
from modules.ui.models.TrainingModel import TrainingModel
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.time_util import get_string_timestamp

import torch


class SamplingModel(SingletonConfigModel):
    def __init__(self):
        super().__init__(SampleConfig.default_values())
        self.model = None
        self.progress_fn = None

    def __update_preview(self, sampler_output: ModelSamplerOutput):
        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            if self.progress_fn is not None:
                self.progress_fn({"data": image})

    def __update_progress(self, progress, max_progress):
        if self.progress_fn is not None:
            self.progress_fn({"value": progress, "max_value": max_progress})

    def __load_model(self, train_config) -> BaseModel:
        model_loader = create.create_model_loader(
            model_type=train_config.model_type,
            training_method=train_config.training_method,
        )

        model_setup = create.create_model_setup(
            model_type=train_config.model_type,
            train_device=torch.device(train_config.train_device),
            temp_device=torch.device(train_config.temp_device),
            training_method=train_config.training_method,
        )

        model_names = train_config.model_names()
        if train_config.continue_last_backup:
            last_backup_path = train_config.get_last_backup_path()

            if last_backup_path:
                if train_config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif train_config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                self.log("info", f"Loading from backup '{last_backup_path}'...")
            else:
                self.log("info", "No backup found, loading without backup...")

        model = model_loader.load(
            model_type=train_config.model_type,
            model_names=model_names,
            weight_dtypes=train_config.weight_dtypes(),
        )
        model.train_config = train_config

        model_setup.setup_optimizations(model, train_config)
        model_setup.setup_train_device(model, train_config)
        model_setup.setup_model(model, train_config)
        model.to(torch.device(train_config.temp_device))

        return model

    def __create_sampler(self, model, train_config):
        return create.create_model_sampler(
            train_device=torch.device(train_config.train_device),
            temp_device=torch.device(train_config.temp_device),
            model=model,
            model_type=train_config.model_type,
            training_method=train_config.training_method,
        )

    def sample(self, progress_fn=None):
        self.progress_fn = progress_fn

        with self.critical_region_read():
            sample = copy.deepcopy(self.config)

        if TrainingModel.instance().training_commands is not None:
            TrainingModel.instance().training_callbacks.set_on_sample_custom(self.__update_preview)
            TrainingModel.instance().training_callbacks.set_on_update_sample_custom_progress(self.__update_progress)

            TrainingModel.instance().training_commands.sample_custom(sample)
        else:
            with StateModel.instance().critical_region_read():
                train_config = TrainConfig.default_values().from_dict(StateModel.instance().config.to_dict())

            train_config.optimizer.optimizer = None
            train_config.ema = EMAMode.OFF

            if self.model is None:
                # lazy initialization
                self.model = self.__load_model(train_config)
                self.model_sampler = self.__create_sampler(self.model, train_config)

            sample.from_train_config(train_config)

            sample_dir = os.path.join(
                train_config.workspace_dir,
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
                image_format=train_config.sample_image_format,
                video_format=train_config.sample_video_format,
                audio_format=train_config.sample_audio_format,
                on_sample=self.__update_preview,
                on_update_progress=self.__update_progress,
            )

            # TODO: Should self.model be garbage collected?
