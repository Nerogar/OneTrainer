import gc
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import Parameter
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
from modules.util import path_util, create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.enum.DataType import DataType
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
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard_subprocess: subprocess.Popen
    tensorboard: SummaryWriter

    def __init__(self, args: TrainArgs, callbacks: TrainCallbacks, commands: TrainCommands):
        super(GenericTrainer, self).__init__(args, callbacks, commands)

        tensorboard_log_dir = os.path.join(args.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, self.__get_string_timestamp()))
        if args.tensorboard:
            tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")

            self.tensorboard_subprocess = subprocess.Popen(
                [
                    tensorboard_executable,
                    "--logdir",
                    tensorboard_log_dir,
                    "--port",
                    "6006",
                    "--samples_per_plugin=images=100"
                ]
            )
        self.one_step_trained = False

    def start(self):
        if self.args.clear_cache_before_training and self.args.latent_caching:
            self.__clear_cache()

        if self.args.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        self.model = self.model_loader.load(
            self.args.model_type,
            self.args.weight_dtype.torch_dtype(),
            self.args.base_model_name,
            self.args.extra_model_name
        )

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_train_device(self.model, self.args)
        self.model_setup.setup_model(self.model, self.args)
        self.model_setup.setup_eval_device(self.model)
        self.__gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = list(self.model_setup.create_parameters(self.model, self.args))

    def __gc(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def __get_string_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.args.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.args.cache_dir):
            for filename in os.listdir(self.args.cache_dir):
                path = os.path.join(self.args.cache_dir, filename)
                if os.path.isdir(path) and filename.startswith('epoch-'):
                    shutil.rmtree(path)

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(self, train_progress: TrainProgress, sample_definitions: list[dict], folder_postfix: str = ""):
        for i, sample_definition in enumerate(sample_definitions):
            safe_prompt = path_util.safe_filename(sample_definition['prompt'])

            sample_dir = os.path.join(
                self.args.workspace_dir,
                "samples",
                f"{str(i)} - {safe_prompt}{folder_postfix}",
            )

            sample_path = os.path.join(
                sample_dir,
                f"{self.__get_string_timestamp()}-training-sample-{train_progress.global_step}-{train_progress.epoch}-{train_progress.epoch_step}.png"
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
                text_encoder_layer_skip=self.args.text_encoder_layer_skip,
                force_last_timestep=self.args.rescale_noise_scheduler_to_zero_terminal_snr,
                on_sample=on_sample,
            )

            self.__gc()

    def __sample_during_training(self, train_progress: TrainProgress, sample_definitions: list[dict] = None):
        self.__gc()

        self.callbacks.on_update_status("sampling")

        self.model_setup.setup_eval_device(self.model)

        if not sample_definitions:
            with open(self.args.sample_definition_file_name, 'r') as f:
                sample_definitions = json.load(f)

        if self.model.ema:
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(train_progress, sample_definitions)

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if an ema model exists
        if self.model.ema:
            self.__sample_loop(train_progress, sample_definitions, " - no-ema")

        self.model_setup.setup_train_device(self.model, self.args)

        self.__gc()

    def backup(self):
        self.__gc()

        self.callbacks.on_update_status("creating backup")

        backup_path = os.path.join(self.args.workspace_dir, "backup", self.__get_string_timestamp())
        print("Creating Backup " + backup_path)

        try:
            self.model_saver.save(
                self.model,
                self.args.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                torch.float32
            )
        except:
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except:
                print("Could not delete partial backup")
                pass

        self.model_setup.setup_train_device(self.model, self.args)

        self.__gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.action_needed("sample", self.args.sample_after, self.args.sample_after_unit, train_progress)

    def __needs_backup(self, train_progress: TrainProgress):
        return self.action_needed(
            "backup", self.args.backup_after, self.args.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.action_needed(
            "update_step", self.args.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def train(self):
        train_device = torch.device(self.args.train_device)

        train_progress = self.model.train_progress

        if self.args.only_cache:
            self.callbacks.on_update_status("caching")

            self.model_setup.setup_eval_device(self.model)
            self.__gc()

            cached_epochs = [False] * self.args.latent_caching_epochs
            for epoch in tqdm(range(train_progress.epoch, self.args.epochs, 1), desc="epoch"):
                if not cached_epochs[epoch % self.args.latent_caching_epochs]:
                    self.data_loader.ds.start_next_epoch()
                    cached_epochs[epoch % self.args.latent_caching_epochs] = True
            return

        lr_scheduler = create.create_lr_scheduler(
            optimizer=self.model.optimizer,
            learning_rate_scheduler=self.args.learning_rate_scheduler,
            warmup_steps=self.args.learning_rate_warmup_steps,
            num_cycles=self.args.learning_rate_cycles,
            max_epochs=self.args.epochs,
            approximate_epoch_length=self.data_loader.ds.approximate_length(),
            global_step=train_progress.global_step
        )

        if self.args.train_dtype.enable_loss_scaling() and self.args.weight_dtype == DataType.FLOAT_32:
            scaler = GradScaler()
        else:
            scaler = None

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        accumulated_loss = 0.0
        for epoch in tqdm(range(train_progress.epoch, self.args.epochs, 1), desc="epoch"):
            self.callbacks.on_update_status("starting epoch/caching")

            self.model_setup.setup_eval_device(self.model)
            self.__gc()
            self.data_loader.ds.start_next_epoch()
            self.model_setup.setup_train_device(self.model, self.args)
            self.__gc()

            current_epoch_length = len(self.data_loader.dl) + train_progress.epoch_step
            for epoch_step, batch in enumerate(tqdm(self.data_loader.dl, desc="step")):
                if self.__needs_sample(train_progress):
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress)
                    )

                if self.__needs_gc(train_progress):
                    self.__gc()

                sample_command = self.commands.get_and_reset_sample_command()
                if sample_command:
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, [sample_command])
                    )

                if not has_gradient:
                    self.__execute_sample_during_training()

                if self.__needs_backup(train_progress):
                    self.backup()

                self.callbacks.on_update_status("training")

                if self.args.train_dtype.enable_mixed_precision() and self.args.weight_dtype == DataType.FLOAT_32:
                    forward_context = torch.autocast(train_device.type, dtype=self.args.train_dtype.torch_dtype())
                else:
                    forward_context = nullcontext()

                with forward_context:
                    model_output_data = self.model_setup.predict(self.model, batch, self.args, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.args)

                loss = loss / self.args.gradient_accumulation_steps
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                has_gradient = True
                accumulated_loss += loss.item()

                if self.__is_update_step(train_progress):
                    if scaler:
                        scaler.unscale_(self.model.optimizer)
                        nn.utils.clip_grad_norm_(self.parameters, 1)
                        scaler.step(self.model.optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.parameters, 1)
                        self.model.optimizer.step()

                    self.model.optimizer.zero_grad(set_to_none=True)
                    has_gradient = False
                    lr_scheduler.step()

                    self.tensorboard.add_scalar(
                        "learning_rate", lr_scheduler.get_last_lr()[0], train_progress.global_step
                    )
                    self.tensorboard.add_scalar("loss", accumulated_loss, train_progress.global_step)
                    accumulated_loss = 0.0
                    self.model_setup.after_optimizer_step(self.model, self.args, train_progress)
                    if self.model.ema:
                        self.model.ema.step(
                            self.parameters,
                            train_progress.global_step // self.args.gradient_accumulation_steps
                        )
                    self.one_step_trained = True

                train_progress.next_step(self.args.batch_size)
                self.callbacks.on_update_progress(train_progress, current_epoch_length, self.args.epochs)

                if self.commands.get_stop_command():
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_progress(train_progress, current_epoch_length, self.args.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            if self.args.backup_before_save:
                self.backup()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

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
