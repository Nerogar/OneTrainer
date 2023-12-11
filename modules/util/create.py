from typing import Iterable

import torch
import transformers
from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, UniPCMultistepScheduler, SchedulerMixin
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from modules.dataLoader.StableDiffusionEmbeddingDataLoader import StableDiffusionEmbeddingDataLoader
from modules.dataLoader.StableDiffusionFineTuneDataLoader import StableDiffusionFineTuneDataLoader
from modules.dataLoader.StableDiffusionFineTuneVaeDataLoader import StableDiffusionFineTuneVaeDataLoader
from modules.dataLoader.StableDiffusionXLEmbeddingDataLoader import StableDiffusionXLEmbeddingDataLoader
from modules.dataLoader.StableDiffusionXLFineTuneDataLoader import StableDiffusionXLFineTuneDataLoader
from modules.dataLoader.WuerstchenEmbeddingDataLoader import WuerstchenEmbeddingDataLoader
from modules.dataLoader.WuerstchenFineTuneDataLoader import WuerstchenFineTuneDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionEmbeddingModelLoader import StableDiffusionEmbeddingModelLoader
from modules.modelLoader.StableDiffusionLoRAModelLoader import StableDiffusionLoRAModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelLoader.StableDiffusionXLEmbeddingModelLoader import StableDiffusionXLEmbeddingModelLoader
from modules.modelLoader.StableDiffusionXLLoRAModelLoader import StableDiffusionXLLoRAModelLoader
from modules.modelLoader.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.modelLoader.WuerstchenEmbeddingModelLoader import WuerstchenEmbeddingModelLoader
from modules.modelLoader.WuerstchenLoRAModelLoader import WuerstchenLoRAModelLoader
from modules.modelLoader.WuerstchenModelLoader import WuerstchenModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.StableDiffusionSampler import StableDiffusionSampler
from modules.modelSampler.StableDiffusionVaeSampler import StableDiffusionVaeSampler
from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
from modules.modelSampler.WuerstchenSampler import WuerstchenSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.StableDiffusionEmbeddingModelSaver import StableDiffusionEmbeddingModelSaver
from modules.modelSaver.StableDiffusionLoRAModelSaver import StableDiffusionLoRAModelSaver
from modules.modelSaver.StableDiffusionModelSaver import StableDiffusionModelSaver
from modules.modelSaver.StableDiffusionXLEmbeddingModelSaver import StableDiffusionXLEmbeddingModelSaver
from modules.modelSaver.StableDiffusionXLLoRAModelSaver import StableDiffusionXLLoRAModelSaver
from modules.modelSaver.StableDiffusionXLModelSaver import StableDiffusionXLModelSaver
from modules.modelSaver.WuerstchenEmbeddingModelSaver import WuerstchenEmbeddingModelSaver
from modules.modelSaver.WuerstchenLoRAModelSaver import WuerstchenLoRAModelSaver
from modules.modelSaver.WuerstchenModelSaver import WuerstchenModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.StableDiffusionEmbeddingSetup import StableDiffusionEmbeddingSetup
from modules.modelSetup.StableDiffusionFineTuneSetup import StableDiffusionFineTuneSetup
from modules.modelSetup.StableDiffusionFineTuneVaeSetup import StableDiffusionFineTuneVaeSetup
from modules.modelSetup.StableDiffusionLoRASetup import StableDiffusionLoRASetup
from modules.modelSetup.StableDiffusionXLEmbeddingSetup import StableDiffusionXLEmbeddingSetup
from modules.modelSetup.StableDiffusionXLFineTuneSetup import StableDiffusionXLFineTuneSetup
from modules.modelSetup.StableDiffusionXLLoRASetup import StableDiffusionXLLoRASetup
from modules.modelSetup.WuerstchenEmbeddingSetup import WuerstchenEmbeddingSetup
from modules.modelSetup.WuerstchenFineTuneSetup import WuerstchenFineTuneSetup
from modules.modelSetup.WuerstchenLoRASetup import WuerstchenLoRASetup
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
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenModelLoader()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelLoader()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenLoRAModelLoader()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingModelLoader()


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
            if model_type.is_wuerstchen():
                return WuerstchenModelSaver()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionModelSaver()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelSaver()
            if model_type.is_wuerstchen():
                return WuerstchenLoRAModelSaver()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelSaver()
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingModelSaver()


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
            if model_type.is_wuerstchen():
                return WuerstchenFineTuneSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneVaeSetup(train_device, temp_device, debug_mode)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRASetup(train_device, temp_device, debug_mode)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRASetup(train_device, temp_device, debug_mode)
            if model_type.is_wuerstchen():
                return WuerstchenLoRASetup(train_device, temp_device, debug_mode)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingSetup(train_device, temp_device, debug_mode)


def create_model_sampler(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSampler:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(train_device, temp_device, model, model_type)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(train_device, temp_device, model, model_type)
            if model_type.is_wuerstchen():
                return WuerstchenSampler(train_device, temp_device, model, model_type)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionVaeSampler(train_device, temp_device, model, model_type)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(train_device, temp_device, model, model_type)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(train_device, temp_device, model, model_type)
            if model_type.is_wuerstchen():
                return WuerstchenSampler(train_device, temp_device, model, model_type)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(train_device, temp_device, model, model_type)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(train_device, temp_device, model, model_type)
            if model_type.is_wuerstchen():
                return WuerstchenSampler(train_device, temp_device, model, model_type)


def create_data_loader(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        args: TrainArgs = None,
        train_progress: TrainProgress = TrainProgress(),
):
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneVaeDataLoader(train_device, temp_device, args, model, train_progress)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenFineTuneDataLoader(train_device, temp_device, args, model, train_progress)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingDataLoader(train_device, temp_device, args, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingDataLoader(train_device, temp_device, args, model, train_progress)


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
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                dampening=args.optimizer_dampening if args.optimizer_dampening is not None else 0,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                nesterov=args.optimizer_nesterov if args.optimizer_nesterov is not None else False,
                foreach=args.optimizer_foreach if args.optimizer_foreach is not None else False,
                maximize=args.optimizer_maximize if args.optimizer_maximize is not None else False,
                differentiable=args.optimizer_differentiable if args.optimizer_differentiable is not None else False,
            )

        # SGD_8BIT Optimizer
        case Optimizer.SGD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.SGD8bit(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                dampening=args.optimizer_dampening if args.optimizer_dampening is not None else 0,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                nesterov=args.optimizer_nesterov if args.optimizer_nesterov is not None else False,
            )

        # ADAM Optimizer
        case Optimizer.ADAM:
            optimizer = torch.optim.Adam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                amsgrad=args.optimizer_amsgrad if args.optimizer_amsgrad is not None else False,
                foreach=args.optimizer_foreach if args.optimizer_foreach is not None else False,
                maximize=args.optimizer_maximize if args.optimizer_maximize is not None else False,
                capturable=args.optimizer_capturable if args.optimizer_capturable is not None else False,
                differentiable=args.optimizer_differentiable if args.optimizer_differentiable is not None else False,
                fused=args.optimizer_fused if args.optimizer_fused is not None else False,
            )

        # ADAMW Optimizer
        case Optimizer.ADAMW:
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 1e-2,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                amsgrad=args.optimizer_amsgrad if args.optimizer_amsgrad is not None else False,
                foreach=args.optimizer_foreach if args.optimizer_foreach is not None else False,
                maximize=args.optimizer_maximize if args.optimizer_maximize is not None else False,
                capturable=args.optimizer_capturable if args.optimizer_capturable is not None else False,
                differentiable=args.optimizer_differentiable if args.optimizer_differentiable is not None else False,
                fused=args.optimizer_fused if args.optimizer_fused is not None else False,
            )

        # ADAM_8BIT Optimizer
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else True,
                is_paged=args.optimizer_is_paged if args.optimizer_is_paged is not None else False,
            )

        # ADAMW_8BIT Optimizer
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 1e-2,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else True,
                is_paged=args.optimizer_is_paged if args.optimizer_is_paged is not None else False,
            )

        # ADAGRAD Optimizer
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-10,
                lr_decay=args.optimizer_lr_decay if args.optimizer_lr_decay is not None else 0,
                initial_accumulator_value=args.optimizer_initial_accumulator_value if args.optimizer_initial_accumulator_value is not None else 0,
            )

        # ADAGRAD_8BIT Optimizer
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-10,
                lr_decay=args.optimizer_lr_decay if args.optimizer_lr_decay is not None else 0,
                initial_accumulator_value=args.optimizer_initial_accumulator_value if args.optimizer_initial_accumulator_value is not None else 0,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else True,
            )

        # RMSPROP Optimizer
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                alpha=args.optimizer_alpha if args.optimizer_alpha is not None else 0.99,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                centered=args.optimizer_centered if args.optimizer_centered is not None else False,
            )

        # RMSPROP_8BIT Optimizer
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                alpha=args.optimizer_alpha if args.optimizer_alpha is not None else 0.99,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                centered=args.optimizer_centered if args.optimizer_centered is not None else False,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else True,
            )

        # LION Optimizer
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.99),
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                use_triton=args.optimizer_use_triton if args.optimizer_use_triton is not None else False,
            )

        # LARS Optimizer
        case Optimizer.LARS:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                dampening=args.optimizer_dampening if args.optimizer_dampening is not None else 0,
                nesterov=args.optimizer_nesterov if args.optimizer_nesterov is not None else False,
                max_unorm=args.optimizer_max_unorm if args.optimizer_max_unorm is not None else 0.02,
            )

        # LARS_8BIT Optimizer
        case Optimizer.LARS_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                dampening=args.optimizer_dampening if args.optimizer_dampening is not None else 0,
                nesterov=args.optimizer_nesterov if args.optimizer_nesterov is not None else False,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                max_unorm=args.optimizer_max_unorm if args.optimizer_max_unorm is not None else 0.02,
            )

        # LAMB Optimizer
        case Optimizer.LAMB:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                bias_correction=args.optimizer_bias_correction if args.optimizer_bias_correction is not None else True,
                amsgrad=args.optimizer_amsgrad if args.optimizer_amsgrad is not None else False,
                adam_w_mode=args.optimizer_adam_w_mode if args.optimizer_adam_w_mode is not None else True,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else False,
                max_unorm=args.optimizer_max_unorm if args.optimizer_max_unorm is not None else 1.0,
            )

        # LAMB_8BIT Optimizer
        case Optimizer.LAMB_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB8bit(
                params=parameters,
                lr=args.learning_rate,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                bias_correction=args.optimizer_bias_correction if args.optimizer_bias_correction is not None else True,
                amsgrad=args.optimizer_amsgrad if args.optimizer_amsgrad is not None else False,
                adam_w_mode=args.optimizer_adam_w_mode if args.optimizer_adam_w_mode is not None else True,
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else False,
                max_unorm=args.optimizer_max_unorm if args.optimizer_max_unorm is not None else 1.0,
            )

        # LION_8BIT Optimizer
        case Optimizer.LION_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params=parameters,
                lr=args.learning_rate if args.learning_rate is not None else 0,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                min_8bit_size=args.optimizer_min_8bit_size if args.optimizer_min_8bit_size is not None else 4096,
                percentile_clipping=args.optimizer_percentile_clipping if args.optimizer_percentile_clipping is not None else 100,
                block_wise=args.optimizer_block_wise if args.optimizer_block_wise is not None else True,
                is_paged=args.optimizer_is_paged if args.optimizer_is_paged is not None else False,
            )

        # DADAPT_SGD Optimizer
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0.0,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                log_every=args.optimizer_log_every if args.optimizer_log_every is not None else 0,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                growth_rate=args.optimizer_growth_rate if args.optimizer_growth_rate is not None else float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use if args.optimizer_fsdp_in_use is not None else False,
            )

        # DADAPT_ADAM Optimizer
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                log_every=args.optimizer_log_every if args.optimizer_log_every is not None else 0,
                decouple=args.optimizer_decouple if args.optimizer_decouple is not None else False,
                use_bias_correction=args.optimizer_use_bias_correction if args.optimizer_use_bias_correction is not None else False,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                growth_rate=args.optimizer_growth_rate if args.optimizer_growth_rate is not None else float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use if args.optimizer_fsdp_in_use is not None else False,
            )

        # DADAPT_ADAN Optimizer
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.98,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.92,
                       args.optimizer_beta3 if args.optimizer_beta3 is not None else 0.99),
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0.02,
                no_prox=args.optimizer_no_prox if args.optimizer_no_prox is not None else False,
                log_every=args.optimizer_log_every if args.optimizer_log_every is not None else 0,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                growth_rate=args.optimizer_growth_rate if args.optimizer_growth_rate is not None else float('inf'),
            )

        # DADAPT_ADA_GRAD Optimizer
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=args.learning_rate,
                momentum=args.optimizer_momentum if args.optimizer_momentum is not None else 0,
                log_every=args.optimizer_log_every if args.optimizer_log_every is not None else 0,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0.0,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 0.0,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                growth_rate=args.optimizer_growth_rate if args.optimizer_growth_rate is not None else float('inf'),
            )

        # DADAPT_LION Optimizer
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0.0,
                log_every=args.optimizer_log_every if args.optimizer_log_every is not None else 0,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                fsdp_in_use=args.optimizer_fsdp_in_use if args.optimizer_fsdp_in_use is not None else False,
            )

        # PRODIGY Optimizer
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=args.learning_rate,
                betas=(args.optimizer_beta1 if args.optimizer_beta1 is not None else 0.9,
                       args.optimizer_beta2 if args.optimizer_beta2 is not None else 0.999),
                beta3=args.optimizer_beta3 if args.optimizer_beta3 is not None else None,
                eps=args.optimizer_eps if args.optimizer_eps is not None else 1e-8,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0,
                decouple=args.optimizer_decouple if args.optimizer_decouple is not None else True,
                use_bias_correction=args.optimizer_use_bias_correction if args.optimizer_use_bias_correction is not None else False,
                safeguard_warmup=args.optimizer_safeguard_warmup if args.optimizer_safeguard_warmup is not None else False,
                d0=args.optimizer_d0 if args.optimizer_d0 is not None else 1e-6,
                d_coef=args.optimizer_d_coef if args.optimizer_d_coef is not None else 1.0,
                growth_rate=args.optimizer_growth_rate if args.optimizer_growth_rate is not None else float('inf'),
                fsdp_in_use=args.optimizer_fsdp_in_use if args.optimizer_fsdp_in_use is not None else False,
            )

        # ADAFactor Optimizer
        case Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                params=parameters,
                lr=None if args.optimizer_relative_step == True else args.learning_rate,
                eps=(args.optimizer_eps if args.optimizer_eps is not None else 1e-30,
                     args.optimizer_eps2 if args.optimizer_eps2 is not None else 1e-3),
                clip_threshold=args.optimizer_clip_threshold if args.optimizer_clip_threshold is not None else 1.0,
                decay_rate=args.optimizer_decay_rate if args.optimizer_decay_rate is not None else -0.8,
                beta1=args.optimizer_beta1 if args.optimizer_beta1 is not None else None,
                weight_decay=args.optimizer_weight_decay if args.optimizer_weight_decay is not None else 0.0,
                scale_parameter=args.optimizer_scale_parameter if args.optimizer_scale_parameter is not None else True,
                relative_step=args.optimizer_relative_step if args.optimizer_relative_step is not None else True,
                warmup_init=args.optimizer_warmup_init if args.optimizer_warmup_init is not None else False,
            )
        case Optimizer.PRODIGY:
            optimizer = transformers.Adafactor(
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
        case LearningRateScheduler.ADAFACTOR:
            from transformers.optimization import AdafactorSchedule
            return AdafactorSchedule(
                optimizer,
                initial_lr=optimizer.state_dict()['param_groups'][0]['lr'],
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
