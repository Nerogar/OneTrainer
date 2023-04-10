import time

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from modules.dataLoader.MgdsStableDiffusionDataLoader import MgdsStableDiffusionDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util.CustomGradScaler import CustomGradScaler
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

        self.model = self.model_loader.load(self.args.base_model_name, self.args.model_type, self.args.train_dtype)
        self.model_setup.start_data_loader(self.model)

        self.data_loader = self.create_data_loader(self.model)

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1

    def sample_during_training(self, epoch: int, epoch_step: int, global_step: int):
        self.model_setup.start_eval(self.model)
        self.model_sampler.sample(self.args.sample_prompt, 42, f"debug/samples/training-sample-{global_step}-{epoch}-{epoch_step}.png")
        self.model_setup.start_train(self.model, self.args.train_text_encoder)

    def needs_sample(self, epoch: int, global_step: int):
        sample_after = self.args.sample_after

        match self.args.sample_after_unit:
            case TimeUnit.EPOCH:
                return epoch % int(sample_after) == 0
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

    def train(self):
        parameters = self.model.parameters(self.args)

        optimizer = AdamW(
            params=parameters,
            lr=3e-6,
            weight_decay=1e-2,
            eps=1e-4 if self.args.train_dtype == torch.float16 else 1e-8,
        )

        scaler = CustomGradScaler()

        global_step = 0
        for epoch in tqdm(range(100), desc="epoch"):
            self.model_setup.start_data_loader(self.model)
            self.data_loader.ds.start_next_epoch()
            self.model_setup.start_train(self.model, self.args.train_text_encoder)
            for epoch_step, batch in enumerate(tqdm(self.data_loader.dl, desc="sample")):
                predicted, target = self.model.predict(batch, self.args)

                loss = self.loss(batch, predicted.float(), target.float())

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                if self.needs_sample(epoch, global_step):
                    self.sample_during_training(epoch, epoch_step, global_step)

    def end(self):
        model_saver = self.create_model_saver()
        model_saver.save(self.model, self.args.model_type, self.args.output_model_format, self.args.output_model_destination)
