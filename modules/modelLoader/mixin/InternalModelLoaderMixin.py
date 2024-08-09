import contextlib
import json
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.TrainProgress import TrainProgress

import torch


class InternalModelLoaderMixin(metaclass=ABCMeta):
    def _load_internal_data(
            self,
            model: BaseModel,
            model_name: str,
    ):
        if os.path.exists(os.path.join(model_name, "meta.json")):
            # train progress
            with open(os.path.join(model_name, "meta.json")) as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )

            # optimizer
            with contextlib.suppress(FileNotFoundError):
                model.optimizer_state_dict = torch.load(os.path.join(model_name, "optimizer", "optimizer.pt"))

            # ema
            with contextlib.suppress(FileNotFoundError):
                model.ema_state_dict = torch.load(os.path.join(model_name, "ema", "ema.pt"))

            # meta
            model.train_progress = train_progress
