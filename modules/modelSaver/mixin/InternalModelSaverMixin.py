import json
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel

import torch


class InternalModelSaverMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _save_internal_data(
            self,
            model: BaseModel,
            destination: str,
    ):
        # optimizer
        os.makedirs(os.path.join(destination, "optimizer"), exist_ok=True)
        if isinstance(model.optimizer, list):
            # Handle MuonWithAuxAdam case where optimizer is a list
            # We need to merge state dicts from multiple optimizers into one,
            # ensuring parameter indices are unique across all optimizers.
            all_state = {}
            all_param_groups = []
            param_offset = 0
            for opt in model.optimizer:
                opt_state_dict = opt.state_dict()
                # Determine the number of parameters this optimizer manages to calculate the offset for the next one.
                # Parameter indices in state_dict are 0-based for each optimizer.
                max_indices = [max(g['params']) for g in opt_state_dict['param_groups'] if g.get('params')]
                num_params_in_opt = max(max_indices) + 1 if max_indices else 0

                if num_params_in_opt > 0:
                    # Remap state keys by adding the current offset
                    for state_key, state_value in opt_state_dict['state'].items():
                        all_state[state_key + param_offset] = state_value

                    # Remap param indices in param_groups and add to the global list
                    for group in opt_state_dict['param_groups']:
                        group['params'] = [p_idx + param_offset for p_idx in group['params']]

                all_param_groups.extend(opt_state_dict['param_groups'])

                # Update the offset for the next optimizer
                param_offset += num_params_in_opt

            optimizer_state_dict = {
                'state': all_state,
                'param_groups': all_param_groups,
             }
        else:
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
