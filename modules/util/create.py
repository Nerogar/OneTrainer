from typing import Iterable

import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, UniPCMultistepScheduler
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from modules.dataLoader.MgdsKandinskyFineTuneDataLoader import MgdsKandinskyFineTuneDataLoader
from modules.dataLoader.MgdsStableDiffusionEmbeddingDataLoader import MgdsStableDiffusionEmbeddingDataLoader
from modules.dataLoader.MgdsStableDiffusionFineTuneDataLoader import MgdsStableDiffusionFineTuneDataLoader
from modules.dataLoader.MgdsStableDiffusionFineTuneVaeDataLoader import MgdsStableDiffusionFineTuneVaeDataLoader
from modules.dataLoader.MgdsStableDiffusionXLEmbeddingDataLoader import MgdsStableDiffusionXLEmbeddingDataLoader
from modules.dataLoader.MgdsStableDiffusionXLFineTuneDataLoader import MgdsStableDiffusionXLFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.KandinskyLoRAModelLoader import KandinskyLoRAModelLoader
from modules.modelLoader.KandinskyModelLoader import KandinskyModelLoader
from modules.modelLoader.StableDiffusionEmbeddingModelLoader import StableDiffusionEmbeddingModelLoader
from modules.modelLoader.StableDiffusionLoRAModelLoader import StableDiffusionLoRAModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelLoader.StableDiffusionXLEmbeddingModelLoader import StableDiffusionXLEmbeddingModelLoader
from modules.modelLoader.StableDiffusionXLLoRAModelLoader import StableDiffusionXLLoRAModelLoader
from modules.modelLoader.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.KandinskySampler import KandinskySampler
from modules.modelSampler.StableDiffusionSampler import StableDiffusionSampler
from modules.modelSampler.StableDiffusionVaeSampler import StableDiffusionVaeSampler
from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.KandinskyDiffusionModelSaver import KandinskyModelSaver
from modules.modelSaver.KandinskyLoRAModelSaver import KandinskyLoRAModelSaver
from modules.modelSaver.StableDiffusionEmbeddingModelSaver import StableDiffusionEmbeddingModelSaver
from modules.modelSaver.StableDiffusionLoRAModelSaver import StableDiffusionLoRAModelSaver
from modules.modelSaver.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.modelSaver.StableDiffusionXLEmbeddingModelSaver import StableDiffusionXLEmbeddingModelSaver
from modules.modelSaver.StableDiffusionXLLoRAModelSaver import StableDiffusionXLLoRAModelSaver
from modules.modelSaver.StableDiffusionXLModelSaver import StableDiffusionXLModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.KandinskyFineTuneSetup import KandinskyFineTuneSetup
from modules.modelSetup.KandinskyLoRASetup import KandinskyLoRASetup
from modules.modelSetup.StableDiffusionEmbeddingSetup import StableDiffusionEmbeddingSetup
from modules.modelSetup.StableDiffusionFineTuneSetup import StableDiffusionFineTuneSetup
from modules.modelSetup.StableDiffusionFineTuneVaeSetup import StableDiffusionFineTuneVaeSetup
from modules.modelSetup.StableDiffusionLoRASetup import StableDiffusionLoRASetup
from modules.modelSetup.StableDiffusionXLEmbeddingSetup import StableDiffusionXLEmbeddingSetup
from modules.modelSetup.StableDiffusionXLFineTuneSetup import StableDiffusionXLFineTuneSetup
from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
from modules.module.EMAModule import EMAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.NoiseScheduler import NoiseScheduler
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
            elif model_type.is_stable_diffusion_xl():
                return StableDiffusionXLModelLoader()
            elif model_type.is_kandinsky():
                return KandinskyModelLoader()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelLoader()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelLoader()
            elif model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelLoader()
            elif model_type.is_kandinsky():
                return KandinskyLoRAModelLoader()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelLoader()


def create_model_saver(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLModelSaver()
            if model_type.is_kandinsky():
                return KandinskyModelSaver()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelSaver()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelSaver()
            if model_type.is_kandinsky():
                return KandinskyLoRAModelSaver()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelSaver()


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
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLFineTuneSetup(train_device, temp_device, debug_mode)
            elif model_type.is_kandinsky():
                return KandinskyFineTuneSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneVaeSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRASetup(train_device, temp_device, debug_mode)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRASetup(train_device, temp_device, debug_mode)
            if model_type.is_kandinsky():
                return KandinskyLoRASetup(train_device, temp_device, debug_mode)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingSetup(train_device, temp_device, debug_mode)


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
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(model, model_type, train_device)
            if model_type.is_kandinsky():
                return KandinskySampler(model, model_type, train_device)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionVaeSampler(model, model_type, train_device)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(model, model_type, train_device)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(model, model_type, train_device)
            if model_type.is_kandinsky():
                return KandinskySampler(model, model_type, train_device)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(model, model_type, train_device)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(model, model_type, train_device)


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
            if model_type.is_stable_diffusion_xl():
                return MgdsStableDiffusionXLFineTuneDataLoader(args, model, train_progress)
            elif model_type.is_kandinsky():
                return MgdsKandinskyFineTuneDataLoader(args, model, train_progress)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionFineTuneVaeDataLoader(args, model, train_progress)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionFineTuneDataLoader(args, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return MgdsStableDiffusionXLFineTuneDataLoader(args, model, train_progress)
            if model_type.is_kandinsky():
                return MgdsKandinskyFineTuneDataLoader(args, model, train_progress)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return MgdsStableDiffusionEmbeddingDataLoader(args, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return MgdsStableDiffusionXLEmbeddingDataLoader(args, model, train_progress)


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
                foreach=False,  # disabled, because it uses too much VRAM
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
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                use_bias_correction=True,
                safeguard_warmup=True,
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
        num_epochs: int,
        batch_size: int,
        approximate_epoch_length: int,
        gradient_accumulation_steps: int,
        global_step: int = 0,
) -> LRScheduler:
    steps_per_epoch = approximate_epoch_length / batch_size
    total_steps = int(steps_per_epoch * num_epochs / gradient_accumulation_steps)
    warmup_steps = int(warmup_steps / gradient_accumulation_steps)
    scheduler_steps = total_steps - warmup_steps

    match learning_rate_scheduler:
        case LearningRateScheduler.CONSTANT:
            lr_lambda = lr_lambda_constant()

        case LearningRateScheduler.LINEAR:
            lr_lambda = lr_lambda_linear(
                scheduler_steps
            )

        case LearningRateScheduler.COSINE:
            lr_lambda = lr_lambda_cosine(
                scheduler_steps
            )

        case LearningRateScheduler.COSINE_WITH_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_restarts(
                scheduler_steps, num_cycles
            )

        case LearningRateScheduler.COSINE_WITH_HARD_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_hard_restarts(
                scheduler_steps, num_cycles
            )
        case LearningRateScheduler.REX:
            lr_lambda = lr_lambda_rex(
                scheduler_steps
            )
        case _:
            lr_lambda = lr_lambda_constant()

    if warmup_steps > 0:
        lr_lambda = lr_lambda_warmup(warmup_steps, lr_lambda)

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
        last_epoch=int(global_step / gradient_accumulation_steps) - 1,
    )


def create_noise_scheduler(
        noise_scheduler: NoiseScheduler,
        num_inference_timesteps: int,
        num_train_timesteps: int = 1000,
):
    scheduler = None

    match noise_scheduler:
        case NoiseScheduler.DDIM:
            scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
            )
        case NoiseScheduler.EULER:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_A:
            scheduler = EulerAncestralDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
            )
        case NoiseScheduler.DPMPP:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=0,
                prediction_type="epsilon",
                use_karras_sigmas=False,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=0,
                prediction_type="epsilon",
                use_karras_sigmas=False,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_KARRAS:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=True,
            )
        case NoiseScheduler.DPMPP_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC_KARRAS:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                trained_betas=None,
                steps_offset=1,
                prediction_type="epsilon",
                use_karras_sigmas=True,
            )

    if scheduler:
        scheduler.set_timesteps(num_inference_timesteps)

    return scheduler