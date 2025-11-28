import datetime
import time
import traceback

from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.ui.models.StateModel import StateModel
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch


class TrainingModel(SingletonConfigModel):
    def __init__(self):
        super().__init__(None)
        self.training_commands = None
        self.training_callbacks = None
        self.train_config = None

    def backup_now(self):
        with self.critical_region_read():
            if self.training_commands:
                self.training_commands.backup()

    def save_now(self):
        with self.critical_region_read():
            if self.training_commands:
                self.training_commands.save()

    def sample_now(self):
        with self.critical_region_read():
            if self.training_commands:
                self.training_commands.sample_default()

    def train(self, reattach=False, progress_fn=None):
        self.progress_fn = progress_fn

        cfg = StateModel.instance().bulk_read("tensorboard", "tensorboard_always_on",
                                              "cloud.enabled",
                                              as_dict=True)


        StateModel.instance().save_default()

        if cfg["tensorboard"] and not cfg["tensorboard_always_on"] and StateModel.instance().tensorboard_subprocess is not None:
            StateModel.instance().stop_tensorboard()

        with self.critical_region_write():
            self.training_commands = TrainCommands()

            self.training_callbacks = TrainCallbacks(
                on_update_train_progress=self.__on_update_train_progress(),
                on_update_status=self.__on_update_status(),
            )

            with StateModel.instance().critical_region_read():
                self.train_config = TrainConfig.default_values().from_dict(StateModel.instance().config.to_dict())

            torch_gc()

            error_caught = False


            trainer = create.create_trainer(self.train_config, self.training_callbacks, self.training_commands,
                                            reattach=reattach)
        try:
            trainer.start()
            if cfg["cloud.enabled"]:
                StateModel.instance().set_state("secrets.cloud", self.train_config.secrets.cloud)
            self.start_time = time.monotonic()
            trainer.train()
        except Exception:
            if cfg["cloud.enabled"]:
                StateModel.instance().set_state("secrets.cloud", self.train_config.secrets.cloud)
            error_caught = True
            self.log("critical", traceback.format_exc())

        trainer.end()

        # clear gpu memory
        del trainer

        self.training_thread = None
        with self.critical_region_write():
            self.training_commands = None
        torch.clear_autocast_cache()
        torch_gc()

        if error_caught:
            if self.progress_fn is not None:
                self.progress_fn({"status": "Error: check the console for details", "event": "cancelled"})
        else:
            if self.progress_fn is not None:
                self.progress_fn({"status": "Stopped", "event": "finished"})

        if cfg["tensorboard_always_on"] and StateModel.instance().tensorboard_subprocess is not None:
            StateModel.instance().start_tensorboard()


    def __on_update_train_progress(self):
        def f(train_progress: TrainProgress, max_steps: int, max_epoch: int):
            if self.progress_fn is not None:
                self.progress_fn({"epoch": train_progress.epoch,
                    "max_epochs": max_epoch,
                    "step": train_progress.epoch_step,
                    "max_steps": max_steps,
                    "eta": self.__calculate_eta_string(train_progress, max_steps, max_epoch)})
        return f

    def __on_update_status(self):
        def f(status: str):
            if self.progress_fn is not None:
                self.progress_fn({"status": status})
        return f

    def __calculate_eta_string(self, train_progress: TrainProgress, max_step: int, max_epoch: int) -> str | None:
        spent_total = time.monotonic() - self.start_time
        steps_done = train_progress.epoch * max_step + train_progress.epoch_step
        remaining_steps = (max_epoch - train_progress.epoch - 1) * max_step + (max_step - train_progress.epoch_step)
        total_eta = spent_total / steps_done * remaining_steps

        if train_progress.global_step <= 30:
            return "Estimating ..."

        td = datetime.timedelta(seconds=total_eta)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def stop_training(self):
        if self.progress_fn is not None:
            self.progress_fn({"event": "stopping", "status": "Stopping..."})
        if self.training_commands is not None:
            self.training_commands.stop()
