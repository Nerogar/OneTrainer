import os
import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from modules.dataLoader.MgdsStableDiffusionDataLoader import MgdsStableDiffusionDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TimeUnit import TimeUnit


class FineTuneTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: MgdsStableDiffusionDataLoader
    model_sampler: BaseModelSampler
    model: BaseModel
    previous_sample_time: float

    def __init__(self, args: TrainArgs):
        super(FineTuneTrainer, self).__init__(args=args)

    def start(self):
        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.model = self.model_loader.load(self.args.base_model_name, self.args.model_type)
        self.model_setup.setup_gradients(self.model, 0, self.args)
        self.model_setup.setup_eval_device(self.model)

        self.data_loader = self.create_data_loader(self.model)

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1

    def sample_during_training(self, epoch: int, epoch_step: int, global_step: int):
        self.model_setup.setup_eval_device(self.model)
        sample_path = os.path.join(self.args.sample_dir, f"training-sample-{global_step}-{epoch}-{epoch_step}.png")
        self.model_sampler.sample(self.args.sample_prompt, 42, sample_path)
        self.model_setup.setup_train_device(self.model, self.args)

    def needs_sample(self, epoch: int, epoch_step: int, global_step: int):
        sample_after = self.args.sample_after

        match self.args.sample_after_unit:
            case TimeUnit.EPOCH:
                return epoch % int(sample_after) == 0 and epoch_step == 0
            case TimeUnit.STEP:
                return global_step % int(sample_after) == 0
            case TimeUnit.SECOND:
                seconds_since_previous_sample = time.time() - self.previous_sample_time
                if seconds_since_previous_sample > sample_after:
                    self.previous_sample_time = time.time()
                    return True
                else:
                    return False
            case TimeUnit.MINUTE:
                seconds_since_previous_sample = time.time() - self.previous_sample_time
                if seconds_since_previous_sample > (sample_after * 60):
                    self.previous_sample_time = time.time()
                    return True
                else:
                    return False
            case TimeUnit.HOUR:
                seconds_since_previous_sample = time.time() - self.previous_sample_time
                if seconds_since_previous_sample > (sample_after * 60 * 60):
                    self.previous_sample_time = time.time()
                    return True
                else:
                    return False
            case _:
                return False

    def is_update_step(self, epoch: int, epoch_step: int, global_step: int) -> bool:
        return (global_step + 1) % self.args.gradient_accumulation_steps == 0

    def train(self):
        parameters = self.model.parameters(self.args)

        optimizer = AdamW(
            params=parameters,
            lr=3e-6,
            weight_decay=1e-2,
            eps=1e-8,
        )

        scaler = GradScaler()

        global_step = 0
        for epoch in tqdm(range(self.args.epochs), desc="epoch"):
            self.model_setup.setup_eval_device(self.model)
            self.data_loader.ds.start_next_epoch()
            self.model_setup.setup_train_device(self.model, self.args)
            for epoch_step, batch in enumerate(tqdm(self.data_loader.dl, desc="sample")):
                if self.needs_sample(epoch, epoch_step, global_step):
                    self.sample_during_training(epoch, epoch_step, global_step)

                with torch.autocast(self.args.train_device.type, dtype=self.args.train_dtype):
                    predicted, target = self.model.predict(batch, self.args, global_step)

                    loss = self.loss(batch, predicted.float(), target.float())

                if self.is_update_step(epoch, epoch_step, global_step):
                    optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1)
                if self.is_update_step(epoch, epoch_step, global_step):
                    scaler.step(optimizer)
                scaler.update()

                global_step += 1

    def end(self):
        model_saver = self.create_model_saver()
        model_saver.save(self.model, self.args.model_type, self.args.output_model_format, self.args.output_model_destination, self.args.output_dtype)
