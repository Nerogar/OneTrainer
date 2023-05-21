import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL.Image import Image
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from modules.dataLoader.MgdsStableDiffusionFineTuneDataLoader import MgdsStableDiffusionFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import path_util
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: MgdsStableDiffusionFineTuneDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel
    optimizer: Optimizer
    previous_sample_time: float

    tensorboard_subprocess: subprocess.Popen
    tensorboard: SummaryWriter

    sample_definitions: list[dict]

    def __init__(self, args: TrainArgs, callbacks: TrainCallbacks, commands: TrainCommands):
        super(GenericTrainer, self).__init__(args, callbacks, commands)

        tensorboard_log_dir = os.path.join(args.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = SummaryWriter(tensorboard_log_dir)
        if args.tensorboard:
            tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")

            self.tensorboard_subprocess = subprocess.Popen(
                f"{tensorboard_executable} --logdir {tensorboard_log_dir} --port 6006 --samples_per_plugin=images=100"
            )

        with open(args.sample_definition_file_name, 'r') as f:
            self.sample_definitions = json.load(f)

    def start(self):
        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        self.model = self.model_loader.load(
            self.args.model_type, self.args.base_model_name, self.args.extra_model_name
        )

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_train_device(self.model, self.args)
        self.model_setup.setup_model(self.model, self.args)
        self.model_setup.setup_eval_device(self.model)

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup.get_train_progress(self.model, self.args)
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1

    def __sample_during_training(self, train_progress: TrainProgress, sample_definitions: dict = None):
        self.callbacks.on_update_status("sampling")

        self.model_setup.setup_eval_device(self.model)

        if not sample_definitions:
            sample_definitions = self.sample_definitions

        for i, sample_definition in enumerate(sample_definitions):
            safe_prompt = path_util.safe_filename(sample_definition['prompt'])

            sample_dir = os.path.join(
                self.args.workspace_dir,
                "samples",
                f"{str(i)} - {safe_prompt}",
            )

            sample_path = os.path.join(
                sample_dir,
                f"training-sample-{train_progress.global_step}-{train_progress.epoch}-{train_progress.epoch_step}.png"
            )

            def on_sample(image: Image):
                self.tensorboard.add_image(f"sample{str(i)} - {safe_prompt}", pil_to_tensor(image),
                                           train_progress.global_step)
                self.callbacks.on_sample(image)

            self.model_sampler.sample(
                prompt=sample_definition["prompt"],
                resolution=(sample_definition["height"], sample_definition["width"]),
                seed=sample_definition["seed"],
                destination=sample_path,
                on_sample=on_sample,
            )

        self.model_setup.setup_train_device(self.model, self.args)

    def backup(self):
        self.callbacks.on_update_status("creating backup")

        path = os.path.join(self.args.workspace_dir, "backup", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print("Creating Backup " + path)
        self.model_saver.save(
            self.model,
            self.args.model_type,
            ModelFormat.INTERNAL,
            path,
            torch.float32
        )
        self.model_setup.setup_train_device(self.model, self.args)

    def __needs_sample(self, train_progress: TrainProgress):
        return self.action_needed("sample", self.args.sample_after, self.args.sample_after_unit, train_progress)

    def __needs_backup(self, train_progress: TrainProgress):
        return self.action_needed(
            "backup", self.args.backup_after, self.args.backup_after_unit, train_progress, start_at_zero=False
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.action_needed(
            "update_step", self.args.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def train(self):
        train_device = torch.device(self.args.train_device)

        parameters = self.model_setup.create_parameters(self.model, self.args)

        train_progress = self.model_setup.get_train_progress(self.model, self.args)

        if self.args.only_cache:
            self.callbacks.on_update_status("caching")

            self.model_setup.setup_eval_device(self.model)
            cached_epochs = [False] * self.args.latent_caching_epochs
            for epoch in tqdm(range(train_progress.epoch, self.args.epochs, 1), desc="epoch"):
                if not cached_epochs[epoch % self.args.latent_caching_epochs]:
                    self.data_loader.ds.start_next_epoch()
                    for _ in tqdm(self.data_loader.dl, desc="step"):
                        pass
                    cached_epochs[epoch % self.args.latent_caching_epochs] = True
            return

        optimizer = self.model_setup.create_optimizer(self.model, self.args)

        scaler = GradScaler()

        accumulated_loss = 0.0
        for epoch in tqdm(range(train_progress.epoch, self.args.epochs, 1), desc="epoch"):
            self.callbacks.on_update_status("starting epoch/caching")

            self.model_setup.setup_eval_device(self.model)
            self.data_loader.ds.start_next_epoch()
            self.model_setup.setup_train_device(self.model, self.args)

            current_epoch_length = len(self.data_loader.dl)
            for epoch_step, batch in enumerate(tqdm(self.data_loader.dl, desc="step")):
                if self.__needs_sample(train_progress):
                    self.__sample_during_training(train_progress)

                sample_command = self.commands.get_and_reset_sample_command()
                if sample_command:
                    self.__sample_during_training(train_progress, sample_command)

                if self.__needs_backup(train_progress):
                    self.backup()

                self.callbacks.on_update_status("training")

                with torch.autocast(train_device.type, dtype=self.args.train_dtype.torch_dtype()):
                    predicted, target = self.model_setup.predict(self.model, batch, self.args, train_progress)

                    loss = self.loss(batch, predicted.float(), target.float())

                loss = loss / self.args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                if self.__is_update_step(train_progress):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(parameters, 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    self.tensorboard.add_scalar("loss", accumulated_loss, train_progress.global_step)
                    accumulated_loss = 0.0
                    self.model_setup.after_optimizer_step(self.model, self.args, train_progress)

                train_progress.next_step(self.args.batch_size)
                self.callbacks.on_update_progress(train_progress, current_epoch_length, self.args.epochs)

                if self.commands.get_stop_command():
                    break

            train_progress.next_epoch()
            self.callbacks.on_update_progress(train_progress, current_epoch_length, self.args.epochs)

            if self.commands.get_stop_command():
                break

    def end(self):
        if not self.args.only_cache:
            if self.args.backup_before_save:
                self.backup()

            self.callbacks.on_update_status("saving the final model")

            self.model_saver.save(
                model=self.model,
                model_type=self.args.model_type,
                output_model_format=self.args.output_model_format,
                output_model_destination=self.args.output_model_destination,
                dtype=self.args.output_dtype.torch_dtype()
            )

        self.tensorboard.close()

        if self.args.tensorboard:
            self.tensorboard_subprocess.kill()
