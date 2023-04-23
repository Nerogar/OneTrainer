import os
from datetime import datetime

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from tqdm import tqdm

from modules.dataLoader.MgdsStableDiffusionFineTuneDataLoader import MgdsStableDiffusionFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit


class FineTuneTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: MgdsStableDiffusionFineTuneDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel
    optimizer: Optimizer
    previous_sample_time: float

    def __init__(self, args: TrainArgs):
        super(FineTuneTrainer, self).__init__(args=args)

    def start(self):
        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.model = self.model_loader.load(self.args.model_type, self.args.base_model_name, self.args.extra_model_name)

        self.model_setup.setup_train_device(self.model, self.args)
        self.model_setup.setup_model(self.model, self.args)
        self.model_setup.setup_eval_device(self.model)

        self.data_loader = self.create_data_loader(self.model, self.model_setup.get_train_progress(self.model, self.args))
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1

    def __sample_during_training(self, train_progress: TrainProgress):
        self.model_setup.setup_eval_device(self.model)
        sample_path = os.path.join(self.args.sample_dir, f"training-sample-{train_progress.global_step}-{train_progress.epoch}-{train_progress.epoch_step}.png")
        self.model_sampler.sample(prompt=self.args.sample_prompt, resolution=(self.args.sample_resolution, self.args.sample_resolution), seed=42, destination=sample_path)
        self.model_setup.setup_train_device(self.model, self.args)

    def backup(self):
        path = os.path.join(self.args.backup_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
        return self.action_needed("backup", self.args.backup_after, self.args.backup_after_unit, train_progress, start_at_zero=False)

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.action_needed("update_step", self.args.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False)

    def train(self):
        parameters = self.model_setup.create_parameters(self.model, self.args)

        train_progress = self.model_setup.get_train_progress(self.model, self.args)
        optimizer = self.model_setup.create_optimizer(self.model, self.args)

        scaler = GradScaler()

        for epoch in tqdm(range(train_progress.epoch, self.args.epochs, 1), desc="epoch"):
            self.model_setup.setup_eval_device(self.model)
            self.data_loader.ds.start_next_epoch()
            self.model_setup.setup_train_device(self.model, self.args)
            for epoch_step, batch in enumerate(tqdm(self.data_loader.dl, desc="step")):
                if self.__needs_sample(train_progress):
                    self.__sample_during_training(train_progress)

                if self.__needs_backup(train_progress):
                    self.backup()

                with torch.autocast(self.args.train_device.type, dtype=self.args.train_dtype):
                    predicted, target = self.model_setup.predict(self.model, batch, self.args, train_progress)

                    loss = self.loss(batch, predicted.float(), target.float())

                scaler.scale(loss).backward()

                if self.__is_update_step(train_progress):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(parameters, 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    self.model_setup.after_optimizer_step(self.model, self.args, train_progress)

                train_progress.next_step(self.args.batch_size)

            train_progress.next_epoch()

    def end(self):
        if self.args.backup_before_save:
            self.backup()

        self.model_saver.save(self.model, self.args.model_type, self.args.output_model_format, self.args.output_model_destination, self.args.output_dtype)
