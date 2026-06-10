import contextlib
import json
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.TrainProgress import TrainProgress

import torch


class InternalModelLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _load_internal_data(
            self,
            model: BaseModel,
            model_name: str,
    ):
        if os.path.exists(os.path.join(model_name, "meta.json")):
            # train progress
            with open(os.path.join(model_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )
                if 'last_action_epoch' in meta:
                    train_progress.last_action_epoch = dict(meta['last_action_epoch'])
                elif train_progress.epoch_step > 0:
                    # Legacy backup taken mid-epoch: start-of-epoch actions
                    # already fired in the pre-fix session. Pre-fill markers
                    # to prevent a duplicate fire on the first resumed batch.
                    train_progress.last_action_epoch = {
                        'validate': train_progress.epoch,
                        'sample': train_progress.epoch,
                    }
                else:
                    train_progress.last_action_epoch = {}

            # optimizer
            with contextlib.suppress(FileNotFoundError):
                model.optimizer_state_dict = torch.load(os.path.join(model_name, "optimizer", "optimizer.pt"),
                                                        weights_only=True)

            # ema
            with contextlib.suppress(FileNotFoundError):
                model.ema_state_dict = torch.load(os.path.join(model_name, "ema", "ema.pt"), weights_only=True)

            # meta
            model.train_progress = train_progress
