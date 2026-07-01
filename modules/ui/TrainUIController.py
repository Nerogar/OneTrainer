import datetime
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import scripts.generate_debug_report
from modules.ui.BaseTrainUIView import BaseTrainUIView
from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.ConvertModelUIController import ConvertModelUIController
from modules.ui.SampleWindowController import SampleWindowController
from modules.ui.VideoToolUIController import VideoToolUIController
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui.validation import flush_and_validate_all

import torch


class TrainUIController:
    def __init__(self, config: TrainConfig):
        self.train_config = config
        self.view: BaseTrainUIView | None = None

        self.training_thread = None
        self.training_callbacks: TrainCallbacks | None = None
        self.training_commands: TrainCommands | None = None
        self.always_on_tensorboard_subprocess = None
        self.current_workspace_dir = config.workspace_dir
        self.start_time: float | None = None
        self.start_total_steps: int | None = None

    def on_update_train_progress(self, train_progress: TrainProgress, max_step: int, max_epoch: int):
        # capture session start on first progress update
        if self.start_total_steps is None:
            self.start_total_steps = train_progress.epoch * max_step + train_progress.epoch_step
        eta_str = self._calculate_eta_string(train_progress, max_step, max_epoch)
        self.view.on_update_progress(train_progress.epoch_step, max_step, train_progress.epoch, max_epoch, eta_str)

    def on_update_status(self, status: str):
        self.view.on_update_status(status)

    def _calculate_eta_string(self, train_progress: TrainProgress, max_step: int, max_epoch: int) -> str | None:
        assert self.start_time is not None and self.start_total_steps is not None

        spent_total = time.monotonic() - self.start_time

        # calculate steps done in THIS SESSION only
        current_total_steps = train_progress.epoch * max_step + train_progress.epoch_step
        steps_done_this_session = current_total_steps - self.start_total_steps

        remaining_steps = (max_epoch - train_progress.epoch - 1) * max_step + (max_step - train_progress.epoch_step)

        if steps_done_this_session <= 30:
            return "Estimating ..."

        total_eta = spent_total / steps_done_this_session * remaining_steps

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

    def _check_start_always_on_tensorboard(self):
        if self.train_config.tensorboard_always_on and not self.always_on_tensorboard_subprocess:
            self._start_always_on_tensorboard()

    def _start_always_on_tensorboard(self):
        if self.always_on_tensorboard_subprocess:
            self._stop_always_on_tensorboard()

        tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        tensorboard_log_dir = os.path.join(self.train_config.workspace_dir, "tensorboard")

        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)

        tensorboard_args = [
            tensorboard_executable,
            "--logdir",
            tensorboard_log_dir,
            "--port",
            str(self.train_config.tensorboard_port),
            "--samples_per_plugin=images=100,scalars=10000",
        ]

        if self.train_config.tensorboard_expose:
            tensorboard_args.append("--bind_all")

        try:
            self.always_on_tensorboard_subprocess = subprocess.Popen(tensorboard_args)
        except Exception:
            self.always_on_tensorboard_subprocess = None

    def _stop_always_on_tensorboard(self):
        if self.always_on_tensorboard_subprocess:
            try:
                self.always_on_tensorboard_subprocess.terminate()
                self.always_on_tensorboard_subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.always_on_tensorboard_subprocess.kill()
            except Exception:
                pass
            finally:
                self.always_on_tensorboard_subprocess = None

    def _on_workspace_dir_change(self, new_workspace_dir: str):
        if new_workspace_dir != self.current_workspace_dir:
            self.current_workspace_dir = new_workspace_dir

            if self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._start_always_on_tensorboard()

    def _on_workspace_dir_change_trace(self, *args):
        new_workspace_dir = self.train_config.workspace_dir
        if new_workspace_dir != self.current_workspace_dir:
            self.current_workspace_dir = new_workspace_dir

            if self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._start_always_on_tensorboard()

    def _on_always_on_tensorboard_toggle(self):
        if self.train_config.tensorboard_always_on:
            if not (self.training_thread and self.train_config.tensorboard):
                self._start_always_on_tensorboard()
        else:
            if not (self.training_thread and self.train_config.tensorboard):
                self._stop_always_on_tensorboard()

    def open_tensorboard(self):
        webbrowser.open("http://localhost:" + str(self.train_config.tensorboard_port), new=0, autoraise=False)

    def open_dataset_tool(self, parent, view_cls):
        return CaptionUIController(None, False).create_window(parent, view_cls)

    def open_video_tool(self, parent, view_cls):
        return VideoToolUIController().create_window(parent, view_cls)

    def open_convert_model_tool(self, parent, view_cls):
        return ConvertModelUIController().create_window(parent, view_cls)

    def open_sampling_tool(self, parent, view_cls):
        if not self.training_callbacks and not self.training_commands:
            controller = SampleWindowController(
                self.train_config,
                use_external_model=False,
            )
            window = view_cls(parent, controller)
            parent.show_window(window)
            parent.connect_window_closed(window, torch_gc)

    def open_manual_sample_window(self, parent, view_cls):
        training_callbacks = self.training_callbacks
        training_commands = self.training_commands

        if training_callbacks and training_commands:
            controller = SampleWindowController(
                self.train_config,
                use_external_model=True,
                callbacks=training_callbacks,
                commands=training_commands,
            )
            window = view_cls(parent, controller)
            parent.show_window(window)
            parent.connect_window_closed(window, lambda: training_callbacks.set_on_sample_custom())

    def sample_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.sample_default()

    def backup_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.backup()

    def save_now(self):
        train_commands = self.training_commands
        if train_commands:
            train_commands.save()

    def export_training(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.train_config.to_pack_dict(secrets=False), f, indent=4)

    def generate_debug_package(self, zip_path: Path):
        self.view.on_update_status("Generating debug package...")
        try:
            config_json_string = json.dumps(self.train_config.to_pack_dict(secrets=False))
            scripts.generate_debug_report.create_debug_package(str(zip_path), config_json_string)
            self.view.on_update_status(f"Debug package saved to {zip_path.name}")
        except Exception as e:
            traceback.print_exc()
            self.view.on_update_status(f"Error generating debug package: {e}")

    def __training_thread_function(self):
        error_caught = False

        self.training_callbacks = TrainCallbacks(
            on_update_train_progress=self.on_update_train_progress,
            on_update_status=self.on_update_status,
        )

        trainer = create.create_trainer(self.train_config, self.training_callbacks, self.training_commands, reattach=self.view.get_cloud_reattach())
        try:
            trainer.start()
            if self.train_config.cloud.enabled:
                self.view.sync_cloud_secrets()

            # Reset session tracking - actual values captured on first progress callback
            self.start_total_steps = None
            self.start_time = time.monotonic()
            trainer.train()
        except Exception:
            if self.train_config.cloud.enabled:
                self.view.sync_cloud_secrets()
            error_caught = True
            traceback.print_exc()

        trainer.end()

        # clear gpu memory
        del trainer

        self.training_thread = None
        self.training_commands = None
        torch.clear_autocast_cache()
        torch_gc()

        if error_caught:
            self.on_update_status("Error: check the console for details")
        else:
            self.on_update_status("Stopped")

        # queue UI update on Tk main thread; on_training_stopped applies shared styles, avoid potential race/crash
        self.view.schedule_on_main_thread(lambda: self.view.on_training_stopped(error_caught))

        if self.train_config.tensorboard_always_on and not self.always_on_tensorboard_subprocess:
            self.view.schedule_on_main_thread(self._start_always_on_tensorboard)

    def start_training(self):
        if self.training_thread is None:
            self.view.save_default()

            errors = flush_and_validate_all()
            if errors:
                self.view.show_validation_errors(errors)
                return

            self.view.on_training_started()

            if self.train_config.tensorboard and not self.train_config.tensorboard_always_on and self.always_on_tensorboard_subprocess:
                self._stop_always_on_tensorboard()

            self.training_commands = TrainCommands()
            torch_gc()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            self.view.on_training_stopping()
            self.on_update_status("Stopping ...")
            self.training_commands.stop()
