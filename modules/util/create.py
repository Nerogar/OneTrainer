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
                momentum=args.optimizer_momentum or 0,
                dampening=args.optimizer_dampening or 0,
                weight_decay=args.optimizer_weight_decay or 0,
                nesterov=args.optimizer_nesterov or False,
                foreach=args.optimizer_foreach or False,
                maximize=args.optimizer_maximize or False,
                differentiable=args.optimizer_differentiable or False
            )

        # SGD_8BIT Optimizer
        case Optimizer.SGD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.SGD8bit(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum or 0,
                dampening=args.optimizer_dampening or 0,
                weight_decay=args.optimizer_weight_decay or 0,
                nesterov=args.optimizer_nesterov or False
            )

        # ADAM Optimizer
        case Optimizer.ADAM:
            optimizer = torch.optim.Adam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-8,
                amsgrad=args.optimizer_amsgrad or False,
                foreach=args.optimizer_foreach or False,
                maximize=args.optimizer_maximize or False,
                capturable=args.optimizer_capturable or False,
                differentiable=args.optimizer_differentiable or False,
                fused=args.optimizer_fused or False
            )

        # ADAMW Optimizer
        case Optimizer.ADAMW:
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                weight_decay=args.optimizer_weight_decay or 1e-2,
                eps=args.optimizer_eps or 1e-8,
                amsgrad=args.optimizer_amsgrad or False,
                foreach=args.optimizer_foreach or False,
                maximize=args.optimizer_maximize or False,
                capturable=args.optimizer_capturable or False,
                differentiable=args.optimizer_differentiable or False,
                fused=args.optimizer_fused or False
            )

        # ADAM_8BIT Optimizer
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-8,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or True,
                is_paged=args.optimizer_is_paged or False
            )

        # ADAMW_8BIT Optimizer
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 1e-2,
                eps=args.optimizer_eps or 1e-8,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or True,
                is_paged=args.optimizer_is_paged or False
            )

        # ADAGRAD Optimizer
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-10,
                lr_decay=args.optimizer_lr_decay or 0,
                initial_accumulator_value=args.optimizer_initial_accumulator_value or 0
            )

        # ADAGRAD_8BIT Optimizer
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-10,
                lr_decay=args.optimizer_lr_decay or 0,
                initial_accumulator_value=args.optimizer_initial_accumulator_value or 0,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or True
            )

        # RMSPROP Optimizer
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-8,
                alpha=args.optimizer_alpha or 0.99,
                momentum=args.optimizer_momentum or 0,
                centered=args.optimizer_centered or False
            )

        # RMSPROP_8BIT Optimizer
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                eps=args.optimizer_eps or 1e-8,
                alpha=args.optimizer_alpha or 0.99,
                momentum=args.optimizer_momentum or 0,
                centered=args.optimizer_centered or False,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or True
            )

        # LION Optimizer
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.99),
                weight_decay=args.optimizer_weight_decay or 0,
                use_triton=args.optimizer_use_triton or False
            )

        # LARS Optimizer
        case Optimizer.LARS:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                momentum=args.optimizer_momentum or 0,
                dampening=args.optimizer_dampening or 0,
                nesterov=args.optimizer_nesterov or False,
                max_unorm=args.optimizer_max_unorm or 0.02
            )

        # LARS_8BIT Optimizer
        case Optimizer.LARS_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                momentum=args.optimizer_momentum or 0,
                dampening=args.optimizer_dampening or 0,
                nesterov=args.optimizer_nesterov or False,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                max_unorm=args.optimizer_max_unorm or 0.02
            )

        # LAMB Optimizer
        case Optimizer.LAMB:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                bias_correction=args.optimizer_bias_correction or True,
                amsgrad=args.optimizer_amsgrad or False,
                adam_w_mode=args.optimizer_adam_w_mode or True,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or False,
                max_unorm=args.optimizer_max_unorm or 1.0
            )

        # LAMB_8BIT Optimizer
        case Optimizer.LAMB_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                bias_correction=args.optimizer_bias_correction or True,
                amsgrad=args.optimizer_amsgrad or False,
                adam_w_mode=args.optimizer_adam_w_mode or True,
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or False,
                max_unorm=args.optimizer_max_unorm or 1.0
            )

        # LION_8BIT Optimizer
        case Optimizer.LION_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay or 0,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                min_8bit_size=args.optimizer_min_8bit_size or 4096,
                percentile_clipping=args.optimizer_percentile_clipping or 100,
                block_wise=args.optimizer_block_wise or True,
                is_paged=args.optimizer_is_paged or False
            )

        # DADAPT_SGD Optimizer
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum or 0.0,
                weight_decay=args.optimizer_weight_decay or 0,
                log_every=args.optimizer_log_every or 0,
                d0=args.optimizer_d0 or 1e-6,
                growth_rate=args.optimizer_growth_rate or float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use or False
            )

        # DADAPT_ADAM Optimizer
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                eps=args.optimizer_eps or 1e-8,
                weight_decay=args.optimizer_weight_decay or 0,
                log_every=args.optimizer_log_every or 0,
                decouple=args.optimizer_decouple or False,
                use_bias_correction=args.optimizer_use_bias_correction or False,
                d0=args.optimizer_d0 or 1e-6,
                growth_rate=args.optimizer_growth_rate or float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use or False
            )

        # DADAPT_ADAN Optimizer
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.98, args.optimizer_beta2 or 0.92, args.optimizer_beta3 or 0.99),
                eps=args.optimizer_eps or 1e-8,
                weight_decay=args.optimizer_weight_decay or 0.02,
                no_prox=args.optimizer_no_prox or False,
                log_every=args.optimizer_log_every or 0,
                d0=args.optimizer_d0 or 1e-6,
                growth_rate=args.optimizer_growth_rate or float('inf')
            )

        # DADAPT_ADA_GRAD Optimizer
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum or 0,
                log_every=args.optimizer_log_every or 0,
                weight_decay=args.optimizer_weight_decay or 0.0,
                eps=args.optimizer_eps or 0.0,
                d0=args.optimizer_d0 or 1e-6,
                growth_rate=args.optimizer_growth_rate or float('inf')
            )

        # DADAPT_LION Optimizer
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                weight_decay=args.optimizer_weight_decay or 0.0,
                log_every=args.optimizer_log_every or 0,
                d0=args.optimizer_d0 or 1e-6,
                fsdp_in_use=args.optimizer_fsdp_in_use or False
            )

        # PRODIGY Optimizer
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 or 0.9, args.optimizer_beta2 or 0.999),
                beta3=args.optimizer_beta3 or None,
                eps=args.optimizer_eps or 1e-8,
                weight_decay=args.optimizer_weight_decay or 0,
                decouple=args.optimizer_decouple or True,
                use_bias_correction=args.optimizer_use_bias_correction or False,
                safeguard_warmup=args.optimizer_safeguard_warmup or False,
                d0=args.optimizer_d0 or 1e-6,
                d_coef=args.optimizer_d_coef or 1.0,
                growth_rate=args.optimizer_growth_rate or float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use or False
            )

        # ADAFactor Optimizer
        case Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor
            if args.optimizer_relative_step:
                optimizer = Adafactor(
                    params=parameters,
                    eps=(args.optimizer_eps2 or 1e-3, args.optimizer_eps2 or 1e-3),
                    clip_threshold=args.optimizer_clip_threshold or 1.0,
                    decay_rate=args.optimizer_decay_rate or -0.8,
                    beta1=args.optimizer_beta1 or None,
                    weight_decay=args.optimizer_weight_decay or 0.0,
                    scale_parameter=args.optimizer_scale_parameter or True,
                    relative_step=args.optimizer_relative_step or True,
                    warmup_init=args.optimizer_warmup_init or False
                )
            else:
                optimizer = Adafactor(
                    params=parameters,
                    lr=args.learning_rate,
                    eps=(args.optimizer_eps2 or 1e-3, args.optimizer_eps2 or 1e-3),
                    clip_threshold=args.optimizer_clip_threshold or 1.0,
                    decay_rate=args.optimizer_decay_rate or -0.8,
                    beta1=args.optimizer_beta1 or None,
                    weight_decay=args.optimizer_weight_decay or 0.0,
                    scale_parameter=args.optimizer_scale_parameter or True,
                    relative_step=args.optimizer_relative_step or True,
                    warmup_init=args.optimizer_warmup_init or False
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
    num_train_timesteps = original_noise_scheduler.config.num_train_timesteps if hasattr(
        original_noise_scheduler.config, "num_train_timesteps") else 1000
    beta_start = original_noise_scheduler.config.beta_start if hasattr(original_noise_scheduler.config,
                                                                       "beta_start") else 0.00085
    beta_end = original_noise_scheduler.config.beta_end if hasattr(original_noise_scheduler.config,
                                                                   "beta_end") else 0.012
    beta_schedule = original_noise_scheduler.config.beta_schedule if hasattr(original_noise_scheduler.config,
                                                                             "beta_schedule") else "scaled_linear"
    prediction_type = original_noise_scheduler.config.prediction_type if hasattr(original_noise_scheduler.config,
                                                                                 "prediction_type") else "epsilon"

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
