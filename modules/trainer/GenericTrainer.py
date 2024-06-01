import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import path_util, create
from modules.util.TrainProgress import TrainProgress
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import enable_grad_scaling, create_grad_scaler
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard_subprocess: subprocess.Popen
    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super(GenericTrainer, self).__init__(config, callbacks, commands)

        tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
        os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
        self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, get_string_timestamp()))
        if config.tensorboard:
            tensorboard_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")

            tensorboard_args = [
                tensorboard_executable,
                "--logdir",
                tensorboard_log_dir,
                "--port",
                "6006",
                "--samples_per_plugin=images=100,scalars=10000",
            ]

            if self.config.tensorboard_expose:
                tensorboard_args.append("--bind_all")

            self.tensorboard_subprocess = subprocess.Popen(tensorboard_args)

        self.one_step_trained = False

        self.grad_hook_handles = []

    def start(self):
        if self.config.clear_cache_before_training and self.config.latent_caching:
            self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.__get_last_backup_dirpath()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print(f"No backup found, continuing without backup...")

        self.callbacks.on_update_status("loading the model")
        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    def __get_last_backup_dirpath(self):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            if backup_directories:
                last_backup_dirpath = backup_directories[0]
                return os.path.join(backup_dirpath, last_backup_dirpath)

        return None

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception as e:
                    print(f"Could not delete old rolling backup {dirpath}")

        return None

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig],
            folder_postfix: str = "",
            image_format: ImageFormat = ImageFormat.JPG,
            is_custom_sample: bool = False,
    ):
        for i, sample_params in enumerate(sample_params_list):
            if sample_params.enabled:
                try:
                    safe_prompt = path_util.safe_filename(sample_params.prompt)

                    if is_custom_sample:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            "custom",
                        )
                    else:
                        sample_dir = os.path.join(
                            self.config.workspace_dir,
                            "samples",
                            f"{str(i)} - {safe_prompt}{folder_postfix}",
                        )

                    sample_path = os.path.join(
                        sample_dir,
                        f"{get_string_timestamp()}-training-sample-{train_progress.filename_string()}{image_format.extension()}"
                    )

                    def on_sample_default(image: Image):
                        if self.config.samples_to_tensorboard:
                            self.tensorboard.add_image(f"sample{str(i)} - {safe_prompt}", pil_to_tensor(image),
                                                       train_progress.global_step)
                        self.callbacks.on_sample_default(image)

                    def on_sample_custom(image: Image):
                        self.callbacks.on_sample_custom(image)

                    on_sample = on_sample_custom if is_custom_sample else on_sample_default
                    on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                    self.model.to(self.temp_device)
                    self.model.eval()

                    self.model_sampler.sample(
                        sample_params=sample_params,
                        destination=sample_path,
                        image_format=self.config.sample_image_format,
                        text_encoder_layer_skip=self.config.text_encoder_layer_skip,
                        force_last_timestep=self.config.rescale_noise_scheduler_to_zero_terminal_snr,
                        on_sample=on_sample,
                        on_update_progress=on_update_progress,
                    )
                except:
                    traceback.print_exc()
                    print("Error during sampling, proceeding without sampling")

                torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("sampling")

        is_custom_sample = False
        if not sample_params_list:
            if self.config.samples is not None:
                sample_params_list = self.config.samples
            else:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values().from_dict(samples[i])
                    sample_params_list = samples
        else:
            is_custom_sample = True

        if self.model.ema:
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_params_list=sample_params_list,
            image_format=self.config.sample_image_format,
            is_custom_sample=is_custom_sample,
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if an ema model exists
        if self.model.ema and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_params_list=sample_params_list,
                image_format=self.config.sample_image_format,
                folder_postfix=" - no-ema",
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        shutil.copy2(self.config.concept_file_name, concepts_path)
        shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def backup(self, train_progress: TrainProgress):
        torch_gc()

        self.callbacks.on_update_status("creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            print("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except:
                traceback.print_exc()
                print("Could not delete partial backup")
                pass
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def save(self, train_progress: TrainProgress):
        torch_gc()

        self.callbacks.on_update_status("saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        print("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except:
                traceback.print_exc()
                print("Could not delete partial save")
                pass
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "save", self.config.save_after, self.config.save_after_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        if self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
            if self.config.gradient_accumulation_steps > 1:
                raise RuntimeError("fused_back_step can not be used if gradient_accumulation_steps > 1")

            for param_group in self.model.optimizer.param_groups:
                for i, parameter in enumerate(param_group["params"]):
                    # TODO: Find a better check instead of "parameter.requires_grad".
                    #       This will break if the some parameters don't require grad during the first training step.
                    if parameter.requires_grad:
                        if scaler:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                scaler.unscale_parameter_(tensor, self.model.optimizer)
                                nn.utils.clip_grad_norm_(tensor, 1)
                                scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                                tensor.grad = None
                        else:
                            def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                                nn.utils.clip_grad_norm_(tensor, 1)
                                self.model.optimizer.step_parameter(tensor, param_group, i)
                                tensor.grad = None

                        handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                        self.grad_hook_handles.append(handle)

    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            self.callbacks.on_update_status("caching")
            for epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                self.data_loader.get_data_set().start_next_epoch()
            return

        if enable_grad_scaling(self.config.train_dtype, self.parameters):
            scaler = create_grad_scaler()
        else:
            scaler = None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = 0.0
        ema_loss = None
        for epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
            self.callbacks.on_update_status("starting epoch/caching")

            if self.config.latent_caching:
                self.data_loader.get_data_set().start_next_epoch()
                self.model_setup.setup_train_device(self.model, self.config)
            else:
                self.model_setup.setup_train_device(self.model, self.config)
                self.data_loader.get_data_set().start_next_epoch()

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()
            step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length,
                             initial=train_progress.epoch_step)
            for epoch_step, batch in enumerate(step_tqdm):
                if self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()

                if self.__needs_backup(train_progress) or self.commands.get_and_reset_backup_command():
                    self.backup(train_progress)

                if self.__needs_save(train_progress) or self.commands.get_and_reset_save_command():
                    self.save(train_progress)

                self.callbacks.on_update_status("training")

                model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                loss = loss / self.config.gradient_accumulation_steps
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                has_gradient = True
                accumulated_loss += loss.item()

                if self.__is_update_step(train_progress):
                    if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                        scaler.step_after_unscale_parameter_(self.model.optimizer)
                        scaler.update()
                    elif scaler:
                        scaler.unscale_(self.model.optimizer)
                        nn.utils.clip_grad_norm_(self.parameters, 1)
                        scaler.step(self.model.optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.parameters, 1)
                        self.model.optimizer.step()

                    lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                    self.model.optimizer.zero_grad(set_to_none=True)
                    has_gradient = False

                    self.model_setup.report_to_tensorboard(
                        self.model, self.config, lr_scheduler, self.tensorboard
                    )

                    self.tensorboard.add_scalar("loss/loss", accumulated_loss, train_progress.global_step)
                    ema_loss = ema_loss or accumulated_loss
                    ema_loss = (ema_loss * 0.99) + (accumulated_loss * 0.01)
                    step_tqdm.set_postfix({
                        'loss': accumulated_loss,
                        'smooth loss': ema_loss,
                    })
                    self.tensorboard.add_scalar("loss/smooth loss", ema_loss, train_progress.global_step)
                    accumulated_loss = 0.0

                    self.model_setup.after_optimizer_step(self.model, self.config, train_progress)
                    if self.model.ema:
                        update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                        self.tensorboard.add_scalar(
                            "ema_decay",
                            self.model.ema.get_current_decay(update_step),
                            train_progress.global_step
                        )
                        self.model.ema.step(
                            self.parameters,
                            update_step
                        )

                    self.one_step_trained = True

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            if self.config.backup_before_save:
                self.backup(self.model.train_progress)
            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            self.callbacks.on_update_status("saving the final model")

            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=False)

            print("Saving " + self.config.output_model_destination)

            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=self.config.output_model_destination,
                dtype=self.config.output_dtype.torch_dtype()
            )

        self.tensorboard.close()

        if self.config.tensorboard:
            self.tensorboard_subprocess.kill()

        for handle in self.grad_hook_handles:
            handle.remove()
