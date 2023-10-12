from typing import Iterable

import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, UniPCMultistepScheduler, SchedulerMixin
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
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
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
    
        # SGD Optimizer
        case Optimizer.SGD:
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum,
                dampening=args.optimizer_dampening,
                weight_decay=args.optimizer_weight_decay,
                nesterov=args.optimizer_nesterov,
                foreach=args.optimizer_foreach,
                maximize=args.optimizer_maximize,
                differentiable=args.optimizer_differentiable
            )

        # SGD_8BIT Optimizer
        case Optimizer.SGD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.SGD8bit(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum,
                dampening=args.optimizer_dampening,
                weight_decay=args.optimizer_weight_decay,
                nesterov=args.optimizer_nesterov,
            )

        # ADAM Optimizer
        case Optimizer.ADAM:
            optimizer = torch.optim.Adam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                amsgrad=args.optimizer_amsgrad,
                foreach=args.optimizer_foreach,
                maximize=args.optimizer_maximize,
                capturable=args.optimizer_capturable,
                differentiable=args.optimizer_differentiable,
                fused=args.optimizer_fused
            )

        # ADAMW Optimizer
        case Optimizer.ADAMW:
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                amsgrad=args.optimizer_amsgrad,
                foreach=args.optimizer_foreach,
                maximize=args.optimizer_maximize,
                capturable=args.optimizer_capturable,
                differentiable=args.optimizer_differentiable,
                fused=args.optimizer_fused
            )

        # ADAM_8BIT Optimizer
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
                is_paged=args.optimizer_is_paged
            )

        # ADAMW_8BIT Optimizer
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
                is_paged=args.optimizer_is_paged
            )
            
        # ADAGRAD Optimizer
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                lr_decay=args.optimizer_lr_decay,
                initial_accumulator_value=args.optimizer_initial_accumulator_value,
            )

        # ADAGRAD_8BIT Optimizer
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                lr_decay=args.optimizer_lr_decay,
                initial_accumulator_value=args.optimizer_initial_accumulator_value,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
            )

        # RMSPROP Optimizer
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                alpha=args.optimizer_alpha,
                momentum=args.optimizer_momentum,
                centered=args.optimizer_centered,
            )

        # RMSPROP_8BIT Optimizer
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                alpha=args.optimizer_alpha,
                momentum=args.optimizer_momentum,
                centered=args.optimizer_centered,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
            )
            
        # LION Optimizer
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=args.learning_rate,
                betas=args.optimizer_betas,
                weight_decay=args.optimizer_weight_decay,
                use_triton=args.optimizer_use_triton
            )
            
        # LARS Optimizer
        case Optimizer.LARS:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                momentum=args.optimizer_momentum,
                dampening=args.optimizer_dampening,
                nesterov=args.optimizer_nesterov,
                max_unorm=args.optimizer_max_unorm
            )

        # LARS_8BIT Optimizer
        case Optimizer.LARS_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                momentum=args.optimizer_momentum,
                dampening=args.optimizer_dampening,
                nesterov=args.optimizer_nesterov,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                max_unorm=args.optimizer_max_unorm
            )

        # LAMB Optimizer
        case Optimizer.LAMB:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                bias_correction=args.optimizer_bias_correction,
                amsgrad=args.optimizer_amsgrad,
                adam_w_mode=args.optimizer_adam_w_mode,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
                max_unorm=args.optimizer_max_unorm
            )
            
        # LAMB_8BIT Optimizer
        case Optimizer.LAMB_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                bias_correction=args.optimizer_bias_correction,
                amsgrad=args.optimizer_amsgrad,
                adam_w_mode=args.optimizer_adam_w_mode,
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
                max_unorm=args.optimizer_max_unorm
            )

        # LION_8BIT Optimizer
        case Optimizer.LION_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                min_8bit_size=args.optimizer_min_8bit_size,
                percentile_clipping=args.optimizer_percentile_clipping,
                block_wise=args.optimizer_block_wise,
                is_paged=args.optimizer_is_paged
            )

        # DADAPT_SGD Optimizer
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum,
                dampening=args.optimizer_dampening,
                weight_decay=args.optimizer_weight_decay,
                log_every=args.optimizer_log_every,
                d0=args.optimizer_d0,
                growth_rate=args.optimizer_growth_rate,
                fsdp_in_use=args.optimizer_fsdp_in_use
            )

        # DADAPT_ADAM Optimizer
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                eps=args.optimizer_eps,
                weight_decay=args.optimizer_weight_decay,
                log_every=args.optimizer_log_every,
                decouple=args.optimizer_decouple,
                use_bias_correction=args.optimizer_use_bias_correction,
                d0=args.optimizer_d0,
                growth_rate=args.optimizer_growth_rate,
                fsdp_in_use=args.optimizer_fsdp_in_use
            )

        # DADAPT_ADAN Optimizer
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2, args.optimizer_beta3),
                eps=args.optimizer_eps,
                weight_decay=args.optimizer_weight_decay,
                no_prox=args.optimizer_no_prox,
                log_every=args.optimizer_log_every,
                d0=args.optimizer_d0,
                growth_rate=args.optimizer_growth_rate
            )

        # DADAPT_ADA_GRAD Optimizer
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum,
                log_every=args.optimizer_log_every,
                weight_decay=args.optimizer_weight_decay,
                eps=args.optimizer_eps,
                d0=args.optimizer_d0,
                growth_rate=args.optimizer_growth_rate
            )

        # DADAPT_LION Optimizer
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                weight_decay=args.optimizer_weight_decay,
                log_every=args.optimizer_log_every,
                d0=args.optimizer_d0,
                fsdp_in_use=args.optimizer_fsdp_in_use
            )


        # PRODIGY Optimizer
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1, args.optimizer_beta2),
                beta3=args.optimizer_beta3,
                eps=args.optimizer_eps,
                weight_decay=args.optimizer_weight_decay,
                decouple=args.optimizer_decouple,
                use_bias_correction=args.optimizer_use_bias_correction,
                safeguard_warmup=args.optimizer_safeguard_warmup,
                d0=args.optimizer_d0,
                d_coef=args.optimizer_d_coef,
                growth_rate=args.optimizer_growth_rate,
                fsdp_in_use=args.optimizer_fsdp_in_use
            )
            
        # ADAFactor Optimizer
        case Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor
            if args.optimizer_relative_step:
                optimizer = Adafactor(
                    params=parameters,
                    eps=(args.optimizer_eps2,args.optimizer_eps2),
                    clip_threshold=args.optimizer_clip_threshold,
                    decay_rate=args.optimizer_decay_rate,
                    beta1=args.optimizer_beta1,
                    weight_decay=args.optimizer_weight_decay,
                    scale_parameter=args.optimizer_scale_parameter,
                    relative_step=args.optimizer_relative_step,
                    warmup_init=args.optimizer_warmup_init
                )
            else:
                optimizer = Adafactor(
                    params=parameters,
                    lr=args.learning_rate,
                    eps=(args.optimizer_eps2,args.optimizer_eps2),
                    clip_threshold=args.optimizer_clip_threshold,
                    decay_rate=args.optimizer_decay_rate,
                    beta1=args.optimizer_beta1,
                    weight_decay=args.optimizer_weight_decay,
                    scale_parameter=args.optimizer_scale_parameter,
                    relative_step=args.optimizer_relative_step,
                    warmup_init=args.optimizer_warmup_init
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
        original_noise_scheduler: SchedulerMixin = None,
        num_inference_timesteps: int = None,
):
    scheduler = None

    num_inference_timesteps = num_inference_timesteps or 20
    num_train_timesteps = original_noise_scheduler.config.num_train_timesteps if hasattr(original_noise_scheduler.config, "num_train_timesteps") else 1000
    beta_start = original_noise_scheduler.config.beta_start if hasattr(original_noise_scheduler.config, "beta_start") else 0.00085
    beta_end = original_noise_scheduler.config.beta_end if hasattr(original_noise_scheduler.config, "beta_end") else 0.012
    beta_schedule = original_noise_scheduler.config.beta_schedule if hasattr(original_noise_scheduler.config, "beta_schedule") else "scaled_linear"
    prediction_type = original_noise_scheduler.config.prediction_type if hasattr(original_noise_scheduler.config, "prediction_type") else "epsilon"

    match noise_scheduler:
        case NoiseScheduler.DDIM:
            scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type=prediction_type,
            )
        case NoiseScheduler.EULER:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_A:
            scheduler = EulerAncestralDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
            )
        case NoiseScheduler.DPMPP:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=0,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=0,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=False,
            )
        case NoiseScheduler.EULER_KARRAS:
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
            )
        case NoiseScheduler.DPMPP_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
        case NoiseScheduler.DPMPP_SDE_KARRAS:
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
        case NoiseScheduler.UNIPC_KARRAS:
            scheduler = UniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                trained_betas=None,
                steps_offset=1,
                prediction_type=prediction_type,
                use_karras_sigmas=True,
            )

    if scheduler:
        scheduler.set_timesteps(num_inference_timesteps)

    return scheduler
