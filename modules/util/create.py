from typing import Iterable

import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from modules.dataLoader.MgdsKandinskyFineTuneDataLoader import MgdsKandinskyFineTuneDataLoader
from modules.dataLoader.MgdsStableDiffusionEmbeddingDataLoader import MgdsStableDiffusionEmbeddingDataLoader
from modules.dataLoader.MgdsStableDiffusionFineTuneDataLoader import MgdsStableDiffusionFineTuneDataLoader
from modules.dataLoader.MgdsStableDiffusionFineTuneVaeDataLoader import MgdsStableDiffusionFineTuneVaeDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.KandinskyLoRAModelLoader import KandinskyLoRAModelLoader
from modules.modelLoader.KandinskyModelLoader import KandinskyModelLoader
from modules.modelLoader.StableDiffusionEmbeddingModelLoader import StableDiffusionEmbeddingModelLoader
from modules.modelLoader.StableDiffusionLoRAModelLoader import StableDiffusionLoRAModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.KandinskySampler import KandinskySampler
from modules.modelSampler.StableDiffusionSampler import StableDiffusionSampler
from modules.modelSampler.StableDiffusionVaeSampler import StableDiffusionVaeSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.KandinskyDiffusionModelSaver import KandinskyModelSaver
from modules.modelSaver.KandinskyLoRAModelSaver import KandinskyLoRAModelSaver
from modules.modelSaver.StableDiffusionEmbeddingModelSaver import StableDiffusionEmbeddingModelSaver
from modules.modelSaver.StableDiffusionLoRAModelSaver import StableDiffusionLoRAModelSaver
from modules.modelSaver.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.KandinskyFineTuneSetup import KandinskyFineTuneSetup
from modules.modelSetup.KandinskyLoRASetup import KandinskyLoRASetup
from modules.modelSetup.StableDiffusionEmbeddingSetup import StableDiffusionEmbeddingSetup
from modules.modelSetup.StableDiffusionFineTuneSetup import StableDiffusionFineTuneSetup
from modules.modelSetup.StableDiffusionFineTuneVaeSetup import StableDiffusionFineTuneVaeSetup
from modules.modelSetup.StableDiffusionLoRASetup import StableDiffusionLoRASetup
from modules.module.EMAModule import EMAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.lr_scheduler_util import *


def create_model_loader(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelLoader()
            elif model_type.is_kandinsky():
                return KandinskyModelLoader()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelLoader()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelLoader()
            if model_type.is_kandinsky():
                return KandinskyLoRAModelLoader()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelLoader()


def create_model_saver(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelSaver()
            if model_type.is_kandinsky():
                return KandinskyModelSaver()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelSaver()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelSaver()
            if model_type.is_kandinsky():
                return KandinskyLoRAModelSaver()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelSaver()


def create_model_setup(
        model_type: ModelType,
        train_device: torch.device,
        temp_device: torch.device,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        debug_mode: bool = False,
) -> BaseModelSetup:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneSetup(train_device, temp_device, debug_mode)
            elif model_type.is_kandinsky():
                return KandinskyFineTuneSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneVaeSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRASetup(train_device, temp_device, debug_mode)
            if model_type.is_kandinsky():
                return KandinskyLoRASetup(train_device, temp_device, debug_mode)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingSetup(train_device, temp_device, debug_mode)


def create_model_sampler(
        model: BaseModel,
        model_type: ModelType,
        train_device: torch.device,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSampler:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(model, model_type, train_device)
            if model_type.is_kandinsky():
                return KandinskySampler(model, model_type, train_device)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionVaeSampler(model, model_type, train_device)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(model, model_type, train_device)
            if model_type.is_kandinsky():
                return KandinskySampler(model, model_type, train_device)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(model, model_type, train_device)


def create_data_loader(
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        args: TrainArgs = None,
        train_progress: TrainProgress = TrainProgress(),
):
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionFineTuneDataLoader(args, model, train_progress)
            elif model_type.is_kandinsky():
                return MgdsKandinskyFineTuneDataLoader(args, model, train_progress)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionFineTuneVaeDataLoader(args, model, train_progress)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionFineTuneDataLoader(args, model, train_progress)
            if model_type.is_kandinsky():
                return MgdsKandinskyFineTuneDataLoader(args, model, train_progress)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionEmbeddingDataLoader(args, model, train_progress)


def create_optimizer(
        parameters: Iterable[Parameter] | list[dict],
        state_dict: dict | None,
        args: TrainArgs,
) -> torch.optim.Optimizer:
    optimizer = None

    match args.optimizer:
        case Optimizer.SGD:
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.ADAM:
            optimizer = torch.optim.Adam(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
                foreach=False,  # disabled, because it uses too much VRAM
                fused=True,
            )
        case Optimizer.ADAMW:
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
                foreach=False,  # disabled, because it uses too much VRAM
                fused=True,
            )
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )

    if state_dict is not None:
        for i, params in enumerate(parameters):
            state_dict['param_groups'][i]['lr'] = params['lr']
            state_dict['param_groups'][i]['initial_lr'] = params['initial_lr']

        # TODO: this will break if the optimizer class changed during a restart
        optimizer.load_state_dict(state_dict)

    return optimizer


def create_ema(
        parameters: Iterable[Parameter] | list[dict],
        state_dict: dict | None,
        args: TrainArgs,
) -> EMAModuleWrapper | None:
    if args.ema == EMAMode.GPU:
        device = torch.device(args.train_device)
    elif args.ema == EMAMode.CPU:
        device = torch.device("cpu")
    else:
        return None

    ema = EMAModuleWrapper(
        parameters=parameters,
        decay=args.ema_decay,
        update_step_interval=args.ema_update_step_interval,
        device=device,
    )

    if state_dict is not None:
        ema.load_state_dict(state_dict)

    return ema


def create_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: LearningRateScheduler,
        warmup_steps: int,
        num_cycles: float,
        max_epochs: int,
        approximate_epoch_length: int,
        global_step: int = 0,
) -> LRScheduler:
    match learning_rate_scheduler:
        case LearningRateScheduler.CONSTANT:
            lr_lambda = lr_lambda_constant()

        case LearningRateScheduler.LINEAR:
            lr_lambda = lr_lambda_linear(
                warmup_steps, max_epochs, approximate_epoch_length
            )

        case LearningRateScheduler.COSINE:
            lr_lambda = lr_lambda_cosine(
                warmup_steps, max_epochs, approximate_epoch_length
            )

        case LearningRateScheduler.COSINE_WITH_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_restarts(
                warmup_steps, num_cycles, max_epochs, approximate_epoch_length
            )

        case LearningRateScheduler.COSINE_WITH_HARD_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_hard_restarts(
                warmup_steps, num_cycles, max_epochs, approximate_epoch_length
            )
        case _:
            lr_lambda = lr_lambda_constant()

    if warmup_steps > 0:
        lr_lambda = lr_lambda_warmup(warmup_steps, lr_lambda)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=global_step - 1,
    )
