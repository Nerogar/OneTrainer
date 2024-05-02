import json
import os
from abc import ABCMeta

import torch

from modules.model.BaseModel import BaseModel


class InternalModelSaverMixin(metaclass=ABCMeta):

    def _save_internal_data(
            self,
            model: BaseModel,
            destination: str,
    ):
        # optimizer
        os.makedirs(os.path.join(destination, "optimizer"), exist_ok=True)
        optimizer_state_dict = model.optimizer.state_dict()
        optimizer_state_dict["param_group_mapping"] = model.param_group_mapping
        optimizer_state_dict["param_group_optimizer_mapping"] = \
            [str(model.train_config.optimizer.optimizer) for _ in model.param_group_mapping]

        torch.save(optimizer_state_dict, os.path.join(destination, "optimizer", "optimizer.pt"))

        # ema
        if model.ema:
            os.makedirs(os.path.join(destination, "ema"), exist_ok=True)
            torch.save(model.ema.state_dict(), os.path.join(destination, "ema", "ema.pt"))

        # meta
        with open(os.path.join(destination, "meta.json"), "w") as meta_file:
            json.dump({
                'train_progress': {
                    'epoch': model.train_progress.epoch,
                    'epoch_step': model.train_progress.epoch_step,
                    'epoch_sample': model.train_progress.epoch_sample,
                    'global_step': model.train_progress.global_step,
                },
            }, meta_file)
