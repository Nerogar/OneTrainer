import ast
import importlib
from collections.abc import Iterable

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.EMAModule import EMAModuleWrapper
from modules.util import factory
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.lr_scheduler_util import (
    lr_lambda_constant,
    lr_lambda_cosine,
    lr_lambda_cosine_with_hard_restarts,
    lr_lambda_cosine_with_restarts,
    lr_lambda_linear,
    lr_lambda_rex,
    lr_lambda_warmup,
)
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer.adafactor_extensions import patch_adafactor
from modules.util.optimizer.adam_extensions import patch_adam
from modules.util.optimizer.adamw_extensions import patch_adamw
from modules.util.optimizer.muon_util import split_parameters_for_muon
from modules.util.TrainProgress import TrainProgress
from modules.zluda import ZLUDA

import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    SchedulerMixin,
    UniPCMultistepScheduler,
)

factory.import_dir("modules/modelSampler", "modules.modelSampler")
factory.import_dir("modules/modelLoader", "modules.modelLoader")
factory.import_dir("modules/modelSaver", "modules.modelSaver")
factory.import_dir("modules/modelSetup", "modules.modelSetup")
factory.import_dir("modules/dataLoader", "modules.dataLoader")

def create_model_loader(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader | None:
    cls = factory.get(BaseModelLoader, model_type, training_method)
    return cls() if cls is not None else None


def create_model_saver(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver | None:
    cls = factory.get(BaseModelSaver, model_type, training_method)
    return cls() if cls is not None else None

def get_model_setup_class(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> type | None:
    return factory.get(BaseModelSetup, model_type, training_method)

def create_model_setup(
        model_type: ModelType,
        train_device: torch.device,
        temp_device: torch.device,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        debug_mode: bool = False,
) -> BaseModelSetup | None:
    cls = factory.get(BaseModelSetup, model_type, training_method)
    return cls(train_device, temp_device, debug_mode) if cls is not None else None

def create_model_sampler(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSampler:
    cls = factory.get(BaseModelSampler, model_type, training_method)
    if cls is None:
        cls = factory.get(BaseModelSampler, model_type)
    return cls(train_device, temp_device, model, model_type) if cls is not None else None

def create_data_loader(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        model_setup: BaseModelSetup,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        config: TrainConfig = None,
        train_progress: TrainProgress | None = None,
        is_validation: bool = False
) -> BaseDataLoader | None:
    if config.gradient_checkpointing.offload() and config.layer_offload_fraction > 0 and config.dataloader_threads > 1:
        raise RuntimeError('layer offloading can not be activated if "dataloader_threads" > 1')

    if train_progress is None:
        train_progress = TrainProgress()

    cls = factory.get(BaseDataLoader, model_type, training_method)
    if cls is None:
        cls = factory.get(BaseDataLoader, model_type)
    return cls(train_device, temp_device, config, model, model_setup, train_progress, is_validation) if cls is not None else None

def create_optimizer(
        parameter_group_collection: NamedParameterGroupCollection,
        state_dict: dict | None,
        config: TrainConfig,
        layer_key_fn: dict[int, str] | None = None,
) -> torch.optim.Optimizer | None:
    optimizer = None
    optimizer_config = config.optimizer

    if optimizer_config.optimizer is None:
        return None

    if config.gradient_checkpointing.offload() and config.layer_offload_fraction > 0:
        if (not optimizer_config.optimizer.supports_fused_back_pass() or not optimizer_config.fused_back_pass) \
                and config.training_method == TrainingMethod.FINE_TUNE:
            raise RuntimeError('layer offloading can only be used for fine tuning when using an optimizer that supports "fused_back_pass"')

    parameters = parameter_group_collection.parameters_for_optimizer(config)

    match config.optimizer.optimizer:

        # SGD Optimizer
        case Optimizer.SGD:
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
            )

        # SGD_8BIT Optimizer
        case Optimizer.SGD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.SGD8bit(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
            )

        # ADAM Optimizer
        case Optimizer.ADAM:
            if optimizer_config.stochastic_rounding \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"stochastic_rounding" is only allowed when "fused" and "foreach" are disabled')

            if optimizer_config.fused_back_pass \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"fused_back_pass" is only allowed when "fused" and "foreach" are disabled')

            optimizer = torch.optim.Adam(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                capturable=optimizer_config.capturable if optimizer_config.capturable is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
                fused=optimizer_config.fused if optimizer_config.fused is not None else False,
            )

            if optimizer_config.stochastic_rounding or optimizer_config.fused_back_pass:
                patch_adam(optimizer, optimizer_config.stochastic_rounding)

        # ADAMW Optimizer
        case Optimizer.ADAMW:
            if optimizer_config.stochastic_rounding \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"stochastic_rounding" is only allowed when "fused" and "foreach" are disabled')

            if optimizer_config.fused_back_pass \
                    and (optimizer_config.fused or optimizer_config.foreach):
                raise RuntimeError('"fused_back_pass" is only allowed when "fused" and "foreach" are disabled')

            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False,
                maximize=optimizer_config.maximize if optimizer_config.maximize is not None else False,
                capturable=optimizer_config.capturable if optimizer_config.capturable is not None else False,
                differentiable=optimizer_config.differentiable if optimizer_config.differentiable is not None else False,
                fused=optimizer_config.fused if optimizer_config.fused is not None else False,
            )

            if optimizer_config.stochastic_rounding or optimizer_config.fused_back_pass:
                patch_adamw(optimizer, optimizer_config.stochastic_rounding)

        # ADAM_8BIT Optimizer
        case Optimizer.ADAM_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # ADAMW_8BIT Optimizer
        case Optimizer.ADAMW_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # AdEMAMix_8BIT Optimizer
        case Optimizer.AdEMAMix_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdEMAMix8bit(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta1 is not None else 0.9999,),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # AdEMAMix Optimizer
        case Optimizer.AdEMAMix:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdEMAMix(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta1 is not None else 0.9999,),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                optim_bits=optimizer_config.optim_bits if optimizer_config.optim_bits is not None else 32,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # ADAGRAD Optimizer
        case Optimizer.ADAGRAD:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-10,
                lr_decay=optimizer_config.lr_decay if optimizer_config.lr_decay is not None else 0,
                initial_accumulator_value=optimizer_config.initial_accumulator_value if optimizer_config.initial_accumulator_value is not None else 0,
            )

        # ADAGRAD_8BIT Optimizer
        case Optimizer.ADAGRAD_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adagrad8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-10,
                lr_decay=optimizer_config.lr_decay if optimizer_config.lr_decay is not None else 0,
                initial_accumulator_value=optimizer_config.initial_accumulator_value if optimizer_config.initial_accumulator_value is not None else 0,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
            )

        # RMSPROP Optimizer
        case Optimizer.RMSPROP:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 0.99,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                centered=optimizer_config.centered if optimizer_config.centered is not None else False,
            )

        # RMSPROP_8BIT Optimizer
        case Optimizer.RMSPROP_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.RMSprop8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 0.99,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                centered=optimizer_config.centered if optimizer_config.centered is not None else False,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
            )

        # LION Optimizer
        case Optimizer.LION:
            import lion_pytorch as lp
            optimizer = lp.Lion(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                use_triton=optimizer_config.use_triton if optimizer_config.use_triton is not None else False,
            )

        # LARS Optimizer
        case Optimizer.LARS:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 0.02,
            )

        # LARS_8BIT Optimizer
        case Optimizer.LARS_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LARS8bit(
                params=parameters,
                lr=config.learning_rate,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                dampening=optimizer_config.dampening if optimizer_config.dampening is not None else 0,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else False,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 0.02,
            )

        # LAMB Optimizer
        case Optimizer.LAMB:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB(
                params=parameters,
                lr=config.learning_rate,
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                adam_w_mode=optimizer_config.adam_w_mode if optimizer_config.adam_w_mode is not None else True,
                optim_bits=optimizer_config.optim_bits if optimizer_config.optim_bits is not None else 32,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 1.0,
            )

        # LAMB_8BIT Optimizer
        case Optimizer.LAMB_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.LAMB8bit(
                params=parameters,
                lr=config.learning_rate,
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                adam_w_mode=optimizer_config.adam_w_mode if optimizer_config.adam_w_mode is not None else True,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else False,
                max_unorm=optimizer_config.max_unorm if optimizer_config.max_unorm is not None else 1.0,
            )

        # LION_8BIT Optimizer
        case Optimizer.LION_8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Lion8bit(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 4096,
                percentile_clipping=optimizer_config.percentile_clipping if optimizer_config.percentile_clipping is not None else 100,
                block_wise=optimizer_config.block_wise if optimizer_config.block_wise is not None else True,
                is_paged=optimizer_config.is_paged if optimizer_config.is_paged is not None else False,
            )

        # Schedule-free AdamW
        case Optimizer.SCHEDULE_FREE_ADAMW:
            if config.model_type.is_wuerstchen_v2() or config.model_type.is_stable_cascade():
                raise NotImplementedError("Cannot use schedule-free optimizers with Wuerstchen-based models.")
            from schedulefree import AdamWScheduleFree
            optimizer = AdamWScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 1e-2,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                warmup_steps=config.learning_rate_warmup_steps,
                r=optimizer_config.r if optimizer_config.r is not None else 0,
                weight_lr_power=optimizer_config.weight_lr_power if optimizer_config.weight_lr_power is not None else 2.0,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False
            )

        # Schedule-free SGD
        case Optimizer.SCHEDULE_FREE_SGD:
            if config.model_type.is_wuerstchen_v2() or config.model_type.is_stable_cascade():
                raise NotImplementedError("Cannot use schedule-free optimizers with Wuerstchen models.")
            from schedulefree import SGDScheduleFree
            optimizer = SGDScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                warmup_steps=config.learning_rate_warmup_steps,
                r=optimizer_config.r if optimizer_config.r is not None else 0,
                weight_lr_power=optimizer_config.weight_lr_power if optimizer_config.weight_lr_power is not None else 2.0,
                foreach=optimizer_config.foreach if optimizer_config.foreach is not None else False
            )

        # DADAPT_SGD Optimizer
        case Optimizer.DADAPT_SGD:
            import dadaptation as da
            optimizer = da.DAdaptSGD(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0.0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # DADAPT_ADAM Optimizer
        case Optimizer.DADAPT_ADAM:
            import dadaptation as da
            optimizer = da.DAdaptAdam(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                decouple=optimizer_config.decouple if optimizer_config.decouple is not None else False,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # DADAPT_ADAN Optimizer
        case Optimizer.DADAPT_ADAN:
            import dadaptation as da
            optimizer = da.DAdaptAdan(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.98,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.92,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.02,
                no_prox=optimizer_config.no_prox if optimizer_config.no_prox is not None else False,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
            )

        # DADAPT_ADA_GRAD Optimizer
        case Optimizer.DADAPT_ADA_GRAD:
            import dadaptation as da
            optimizer = da.DAdaptAdaGrad(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 0.0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
            )

        # DADAPT_LION Optimizer
        case Optimizer.DADAPT_LION:
            import dadaptation as da
            optimizer = da.DAdaptLion(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                log_every=optimizer_config.log_every if optimizer_config.log_every is not None else 0,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
            )

        # PRODIGY Optimizer
        case Optimizer.PRODIGY:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                decouple=optimizer_config.decouple if optimizer_config.decouple is not None else True,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                safeguard_warmup=optimizer_config.safeguard_warmup if optimizer_config.safeguard_warmup is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                fsdp_in_use=optimizer_config.fsdp_in_use if optimizer_config.fsdp_in_use is not None else False,
                slice_p=optimizer_config.slice_p if optimizer_config.slice_p is not None else 1,
            )

        # PRODIGY_PLUS_SCHEDULE_FREE Optimizer
        case Optimizer.PRODIGY_PLUS_SCHEDULE_FREE:
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
            optimizer = ProdigyPlusScheduleFree(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decay_by_lr=optimizer_config.weight_decay_by_lr if optimizer_config.weight_decay_by_lr is not None else True,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else False,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                prodigy_steps=optimizer_config.prodigy_steps if optimizer_config.prodigy_steps is not None else 0,
                use_speed=optimizer_config.use_speed if optimizer_config.use_speed is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else None,
                split_groups=optimizer_config.split_groups if optimizer_config.split_groups is not None else True,
                split_groups_mean=optimizer_config.split_groups_mean if optimizer_config.split_groups_mean is not None else True,
                factored=optimizer_config.factored if optimizer_config.factored is not None else True,
                factored_fp32=optimizer_config.factored_fp32 if optimizer_config.factored_fp32 is not None else True,
                fused_back_pass=optimizer_config.fused_back_pass if optimizer_config.fused_back_pass is not None else False,
                use_stableadamw=optimizer_config.use_stableadamw if optimizer_config.use_stableadamw is not None else True,
                use_cautious=optimizer_config.use_cautious if optimizer_config.use_cautious is not None else False,
                use_grams=optimizer_config.use_grams if optimizer_config.use_grams is not None else False,
                use_adopt=optimizer_config.use_adopt if optimizer_config.use_adopt is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding if optimizer_config.stochastic_rounding is not None else True,
                d_limiter=optimizer_config.d_limiter if optimizer_config.d_limiter is not None else True,
                use_schedulefree=optimizer_config.use_schedulefree if optimizer_config.use_schedulefree is not None else True,
                schedulefree_c=optimizer_config.schedulefree_c if optimizer_config.schedulefree_c is not None else 0.0,
                use_orthograd=optimizer_config.use_orthograd if optimizer_config.use_orthograd is not None else False,
            )

        # ADAFactor Optimizer
        case Optimizer.ADAFACTOR:
            from transformers.optimization import Adafactor

            if optimizer_config.relative_step:
                for parameter in parameters:
                    if isinstance(parameter, dict) and 'lr' in parameter:
                        parameter.pop('lr')

            optimizer = Adafactor(
                params=parameters,
                lr=None if optimizer_config.relative_step is True else config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-3),
                clip_threshold=optimizer_config.clip_threshold if optimizer_config.clip_threshold is not None else 1.0,
                decay_rate=optimizer_config.decay_rate if optimizer_config.decay_rate is not None else -0.8,
                beta1=optimizer_config.beta1 if optimizer_config.beta1 is not None else None,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                scale_parameter=optimizer_config.scale_parameter if optimizer_config.scale_parameter is not None else True,
                relative_step=optimizer_config.relative_step if optimizer_config.relative_step is not None else True,
                warmup_init=optimizer_config.warmup_init if optimizer_config.warmup_init is not None else False,
            )

            patch_adafactor(optimizer, optimizer_config.stochastic_rounding)

        # CAME Optimizer
        case Optimizer.CAME:
            from modules.util.optimizer.CAME import CAME
            optimizer = CAME(
                params=parameters,
                lr=config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-16),
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_cautious=optimizer_config.use_cautious,
            )

        # CAME_8BIT Optimizer
        case Optimizer.CAME_8BIT:
            from modules.util.optimizer.CAME8bit import CAME8bit
            optimizer = CAME8bit(
                params=parameters,
                lr=config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-16),
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                min_8bit_size=optimizer_config.min_8bit_size if optimizer_config.min_8bit_size is not None else 16384,
                quant_block_size=optimizer_config.quant_block_size if optimizer_config.quant_block_size is not None else 2048
            )

        # ADAMW_ADV Optimizer
        case Optimizer.ADAMW_ADV:
            from adv_optm import AdamW_adv
            optimizer = AdamW_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else True,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                cautious_mask=optimizer_config.cautious_mask if optimizer_config.cautious_mask is not None else False,
                grams_moment=optimizer_config.grams_moment if optimizer_config.grams_moment is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                use_AdEMAMix=optimizer_config.use_AdEMAMix if optimizer_config.use_AdEMAMix is not None else False,
                beta3_ema=optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=optimizer_config.k_warmup_steps if optimizer_config.k_warmup_steps is not None else 0,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # ADOPT_ADV Optimizer
        case Optimizer.ADOPT_ADV:
            from adv_optm import Adopt_adv
            optimizer = Adopt_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.9999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-6,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                cautious_mask=optimizer_config.cautious_mask if optimizer_config.cautious_mask is not None else False,
                grams_moment=optimizer_config.grams_moment if optimizer_config.grams_moment is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                use_AdEMAMix=optimizer_config.use_AdEMAMix if optimizer_config.use_AdEMAMix is not None else False,
                beta3_ema=optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=optimizer_config.k_warmup_steps if optimizer_config.k_warmup_steps is not None else 0,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # PRODIGY_ADV Optimizer
        case Optimizer.PRODIGY_ADV:
            from adv_optm import Prodigy_adv
            optimizer = Prodigy_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                slice_p=optimizer_config.slice_p if optimizer_config.slice_p is not None else 1,
                prodigy_steps=optimizer_config.prodigy_steps if optimizer_config.prodigy_steps is not None else 0,
                d_limiter=optimizer_config.d_limiter if optimizer_config.d_limiter is not None else False,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                cautious_mask=optimizer_config.cautious_mask if optimizer_config.cautious_mask is not None else False,
                grams_moment=optimizer_config.grams_moment if optimizer_config.grams_moment is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                use_AdEMAMix=optimizer_config.use_AdEMAMix if optimizer_config.use_AdEMAMix is not None else False,
                beta3_ema=optimizer_config.beta3_ema if optimizer_config.beta3_ema is not None else 0.9999,
                alpha=optimizer_config.alpha if optimizer_config.alpha is not None else 5,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=optimizer_config.k_warmup_steps if optimizer_config.k_warmup_steps is not None else 0,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # SIMPLIFIED_AdEMAMix Optimizer
        case Optimizer.SIMPLIFIED_AdEMAMix:
            from adv_optm import Simplified_AdEMAMix
            optimizer = Simplified_AdEMAMix(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.99,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
                beta1_warmup=optimizer_config.beta1_warmup if optimizer_config.beta1_warmup is not None else None,
                min_beta1=optimizer_config.min_beta1 if optimizer_config.min_beta1 is not None else 0.9,
                use_bias_correction=optimizer_config.use_bias_correction if optimizer_config.use_bias_correction is not None else True,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                kourkoutas_beta=optimizer_config.kourkoutas_beta if optimizer_config.kourkoutas_beta is not None else False,
                k_warmup_steps=optimizer_config.k_warmup_steps if optimizer_config.k_warmup_steps is not None else 0,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # SignSGD_ADV Optimizer
        case Optimizer.SIGNSGD_ADV:
            from adv_optm import SignSGD_adv
            optimizer = SignSGD_adv(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
            )

        # Stiefel_LoRA Optimizer
        case Optimizer.Stiefel_LoRA:
            from adv_optm import Stiefel_LoRA
            optimizer = Stiefel_LoRA(
                params=parameters,
                lr=config.learning_rate,
                momentum=optimizer_config.momentum if optimizer_config.momentum is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
            )

        # LION_ADV Optimizer
        case Optimizer.LION_ADV:
            from adv_optm import Lion_adv
            optimizer = Lion_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                clip_threshold=optimizer_config.clip_threshold if optimizer_config.clip_threshold is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                cautious_mask=optimizer_config.cautious_mask if optimizer_config.cautious_mask is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                kappa_p=optimizer_config.kappa_p if optimizer_config.kappa_p is not None else 1.0,
                auto_kappa_p=optimizer_config.auto_kappa_p if optimizer_config.auto_kappa_p is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # LION_PRODIGY_ADV Optimizer
        case Optimizer.LION_PRODIGY_ADV:
            from adv_optm import Lion_Prodigy_adv
            optimizer = Lion_Prodigy_adv(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                beta3=optimizer_config.beta3 if optimizer_config.beta3 is not None else None,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                clip_threshold=optimizer_config.clip_threshold if optimizer_config.clip_threshold is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                d0=optimizer_config.d0 if optimizer_config.d0 is not None else 1e-6,
                d_coef=optimizer_config.d_coef if optimizer_config.d_coef is not None else 1.0,
                growth_rate=optimizer_config.growth_rate if optimizer_config.growth_rate is not None else float('inf'),
                slice_p=optimizer_config.slice_p if optimizer_config.slice_p is not None else 1,
                prodigy_steps=optimizer_config.prodigy_steps if optimizer_config.prodigy_steps is not None else 0,
                d_limiter=optimizer_config.d_limiter if optimizer_config.d_limiter is not None else False,
                cautious_mask=optimizer_config.cautious_mask if optimizer_config.cautious_mask is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                kappa_p=optimizer_config.kappa_p if optimizer_config.kappa_p is not None else 1.0,
                auto_kappa_p=optimizer_config.auto_kappa_p if optimizer_config.auto_kappa_p is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
            )

        # MUON_ADV Optimizer
        case Optimizer.MUON_ADV:
            import inspect

            from adv_optm import Muon_adv

            params_for_optimizer, MuonWithAuxAdam = split_parameters_for_muon(parameters, layer_key_fn, config)

            # Prepare Adam-specific keyword arguments from the config
            adam_kwargs = {}
            if MuonWithAuxAdam:
                adam_config = optimizer_config.muon_adam_config
                adam_config_dict = adam_config if isinstance(adam_config, dict) else adam_config.to_dict()

                valid_adam_keys = {k for k in inspect.signature(Muon_adv.__init__).parameters if k.startswith('adam_')}
                adam_kwargs = {
                    key: adam_config_dict[key.removeprefix('adam_')]
                    for key in valid_adam_keys
                    if key.removeprefix('adam_') in adam_config_dict and adam_config_dict[key.removeprefix('adam_')] is not None
                }
                # Manually construct adam_betas from beta1 and beta2
                beta1_adam = adam_config_dict.get('beta1')
                beta2_adam = adam_config_dict.get('beta2')
                adam_kwargs['adam_betas'] = (
                    beta1_adam if beta1_adam is not None else 0.9,
                    beta2_adam if beta2_adam is not None else 0.99
                )
            optimizer = Muon_adv(
                params=params_for_optimizer,
                lr=config.learning_rate,
                beta1=optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                ns_steps=optimizer_config.ns_steps if optimizer_config.ns_steps is not None else 5,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                rms_rescaling=optimizer_config.rms_rescaling if optimizer_config.rms_rescaling is not None else True,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else True,
                normuon_variant=optimizer_config.normuon_variant if optimizer_config.normuon_variant is not None else False,
                beta2_normuon=optimizer_config.beta2_normuon if optimizer_config.beta2_normuon is not None else 0.95,
                normuon_eps=optimizer_config.normuon_eps if optimizer_config.normuon_eps is not None else 1e-8,
                low_rank_ortho=optimizer_config.low_rank_ortho if optimizer_config.low_rank_ortho is not None else False,
                ortho_rank=optimizer_config.ortho_rank if optimizer_config.ortho_rank is not None else 128,
                accelerated_ns=optimizer_config.accelerated_ns if optimizer_config.accelerated_ns is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                approx_mars=optimizer_config.approx_mars if optimizer_config.approx_mars is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
                **adam_kwargs
            )

        # ADAMUON_ADV Optimizer
        case Optimizer.ADAMUON_ADV:
            import inspect

            from adv_optm import AdaMuon_adv

            params_for_optimizer, MuonWithAuxAdam = split_parameters_for_muon(parameters, layer_key_fn, config)

            # Prepare Adam-specific keyword arguments from the config
            adam_kwargs = {}
            if MuonWithAuxAdam:
                adam_config = optimizer_config.muon_adam_config
                # Handle both dict (from JSON/Config) and Object (legacy/runtime)
                adam_config_dict = adam_config if isinstance(adam_config, dict) else adam_config.to_dict()

                valid_adam_keys = {k for k in inspect.signature(AdaMuon_adv.__init__).parameters if k.startswith('adam_')}
                adam_kwargs = {
                    key: adam_config_dict[key.removeprefix('adam_')]
                    for key in valid_adam_keys
                    if key.removeprefix('adam_') in adam_config_dict and adam_config_dict[key.removeprefix('adam_')] is not None
                }
                # Manually construct adam_betas from beta1 and beta2
                adam_beta1 = adam_config_dict.get('beta1')
                adam_beta2 = adam_config_dict.get('beta2')
                adam_kwargs['adam_betas'] = (
                    adam_beta1 if adam_beta1 is not None else 0.9,
                    adam_beta2 if adam_beta2 is not None else 0.99
                )
            optimizer = AdaMuon_adv(
                params=params_for_optimizer,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                    optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.99),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
                ns_steps=optimizer_config.ns_steps if optimizer_config.ns_steps is not None else 5,
                rms_rescaling=optimizer_config.rms_rescaling if optimizer_config.rms_rescaling is not None else True,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                nnmf_factor=optimizer_config.nnmf_factor if optimizer_config.nnmf_factor is not None else False,
                cautious_wd=optimizer_config.cautious_wd if optimizer_config.cautious_wd is not None else False,
                stochastic_rounding=optimizer_config.stochastic_rounding,
                nesterov=optimizer_config.nesterov if optimizer_config.nesterov is not None else True,
                use_atan2=optimizer_config.use_atan2 if optimizer_config.use_atan2 is not None else False,
                Simplified_AdEMAMix=optimizer_config.Simplified_AdEMAMix if optimizer_config.Simplified_AdEMAMix is not None else False,
                alpha_grad=optimizer_config.alpha_grad if optimizer_config.alpha_grad is not None else 100,
                low_rank_ortho=optimizer_config.low_rank_ortho if optimizer_config.low_rank_ortho is not None else False,
                ortho_rank=optimizer_config.ortho_rank if optimizer_config.ortho_rank is not None else 128,
                normuon_variant=optimizer_config.normuon_variant if optimizer_config.normuon_variant is not None else False,
                accelerated_ns=optimizer_config.accelerated_ns if optimizer_config.accelerated_ns is not None else False,
                orthogonal_gradient=optimizer_config.orthogonal_gradient if optimizer_config.orthogonal_gradient is not None else False,
                approx_mars=optimizer_config.approx_mars if optimizer_config.approx_mars is not None else False,
                compiled_optimizer=optimizer_config.compile if optimizer_config.compile is not None else False,
                **adam_kwargs
            )

        # MUON Optimizer
        case Optimizer.MUON:

            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

            params_for_optimizer, ___ = split_parameters_for_muon(parameters, layer_key_fn, config)

            final_param_groups  = []
            for group in params_for_optimizer:
                is_muon = group.get('optim_type') == 'muon'

                if is_muon:
                    final_group = {
                        'params': group['params'],
                        'lr': group['lr'],
                        'use_muon': True,
                        'momentum': optimizer_config.momentum if optimizer_config.momentum is not None else 0.95,
                        'weight_decay': optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                    }
                else:  # is adam
                    adam_config = optimizer_config.muon_adam_config
                    if not isinstance(adam_config, dict):
                        adam_config = adam_config.to_dict()

                    beta1 = adam_config.get('beta1')
                    beta2 = adam_config.get('beta2')
                    eps = adam_config.get('eps')
                    weight_decay = adam_config.get('weight_decay')

                    final_group = {
                        'params': group['params'],
                        'lr': group['lr'],
                        'use_muon': False,
                        'betas': (beta1 if beta1 is not None else 0.9,
                                  beta2 if beta2 is not None else 0.95),
                        'eps': eps if eps is not None else 1e-10,
                        'weight_decay': weight_decay if weight_decay is not None else 0.0,
                    }
                final_param_groups.append(final_group)

            OptimizerClass = MuonWithAuxAdam if multi.world_size() > 1 else SingleDeviceMuonWithAuxAdam
            optimizer = OptimizerClass(param_groups=final_param_groups )

            # Add metadata back to the optimizer's param_groups for the framework to use.
            for i, group in enumerate(optimizer.param_groups):
                original_group = params_for_optimizer[i]
                group['initial_lr'] = original_group.get('initial_lr', original_group['lr'])
                group['name'] = original_group.get('name')
                group['optim_type'] = original_group.get('optim_type')


        # ADABELIEF Optimizer
        case Optimizer.ADABELIEF:
            from timm.optim.adabelief import AdaBelief
            optimizer = AdaBelief(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-16,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                decoupled_decay=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                rectify=optimizer_config.rectify if optimizer_config.rectify is not None else True,
                degenerated_to_sgd=optimizer_config.degenerated_to_sgd if optimizer_config.degenerated_to_sgd is not None else True,
            )

        # TIGER Optimizer
        case Optimizer.TIGER:
            from pytorch_optimizer.optimizer.tiger import Tiger
            optimizer = Tiger(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                beta=optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
            )

        # AIDA Optimizer
        case Optimizer.AIDA:
            from pytorch_optimizer.optimizer.aida import Aida
            optimizer = Aida(
                params=parameters,
                lr=config.learning_rate if config.learning_rate is not None else 0,
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                k=optimizer_config.k if optimizer_config.k is not None else 2,
                xi=optimizer_config.xi if optimizer_config.xi is not None else 1e-20,
                rectify=optimizer_config.rectify if optimizer_config.rectify is not None else False,
                n_sma_threshold=optimizer_config.n_sma_threshold if optimizer_config.n_sma_threshold is not None else 5,
                degenerated_to_sgd=optimizer_config.degenerated_to_sgd if optimizer_config.degenerated_to_sgd is not None else True,
                ams_bound=optimizer_config.ams_bound if optimizer_config.ams_bound is not None else False,
                r=optimizer_config.r if optimizer_config.r is not None else 0.95,
                adanorm=optimizer_config.adanorm if optimizer_config.adanorm is not None else False,
                adam_debias=optimizer_config.adam_debias if optimizer_config.adam_debias is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-8,
            )

        # ADOPT Optimizer
        case Optimizer.ADOPT:
            from pytorch_optimizer.optimizer.adopt import ADOPT
            optimizer = ADOPT(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else False,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                cautious=optimizer_config.cautious if optimizer_config.cautious is not None else False,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-6,
            )

        # YOGI Optimizer
        case Optimizer.YOGI:
            from pytorch_optimizer.optimizer.yogi import Yogi
            optimizer = Yogi(
                params=parameters,
                lr=config.learning_rate,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
                weight_decouple=optimizer_config.decoupled_decay if optimizer_config.decoupled_decay is not None else True,
                fixed_decay=optimizer_config.fixed_decay if optimizer_config.fixed_decay is not None else False,
                r=optimizer_config.r if optimizer_config.r is not None else 0.95,
                adanorm=optimizer_config.adanorm if optimizer_config.adanorm is not None else False,
                adam_debias=optimizer_config.adam_debias if optimizer_config.adam_debias is not None else False,
                initial_accumulator=optimizer_config.initial_accumulator if optimizer_config.initial_accumulator is not None else 1e-6,
                eps=optimizer_config.eps if optimizer_config.eps is not None else 1e-3,
            )

    if state_dict is not None and optimizer is not None:
        if 'param_group_mapping' not in state_dict:
            # Old method of loading the optimizer state. This only works if the param groups did not change.
            for i, params in enumerate(parameters):
                state_dict['param_groups'][i]['lr'] = params['lr']
                state_dict['param_groups'][i]['initial_lr'] = params['initial_lr']
        else:
            # New method of loading the optimizer state. Each group is mapped by a unique name.
            old_state = state_dict['state']
            old_param_groups = state_dict['param_groups']
            old_group_mapping = state_dict['param_group_mapping']
            old_group_optimizer_mapping = state_dict['param_group_optimizer_mapping']

            new_param_groups = optimizer.state_dict()['param_groups']
            if config.optimizer.MuonWithAuxAdam:
                new_group_mapping = []
                for group in optimizer.param_groups:
                    original_name = group.get('name')

                    optim_type = group.get('optim_type', 'unknown')
                    unique_name = f"{original_name}_{optim_type}"
                    new_group_mapping.append(unique_name)
            else:
                new_group_mapping = parameter_group_collection.unique_name_mapping

            state = {}
            param_groups = []
            state_index = 0

            for new_group_index, unique_group_name in enumerate(new_group_mapping):
                if (unique_group_name in old_group_mapping and str(config.optimizer.optimizer) ==
                        old_group_optimizer_mapping[old_group_mapping.index(unique_group_name)]):
                    # the group state was saved in state_dict
                    old_group_index = old_group_mapping.index(unique_group_name)
                    new_group = new_param_groups[new_group_index]
                    old_group = old_param_groups[old_group_index]
                    for i, old_state_index in enumerate(old_group['params']):
                        if old_state_index in old_state:
                            state[state_index] = old_state[old_state_index]
                        old_group['params'][i] = state_index
                        state_index += 1
                    param_groups.append(old_group)

                    old_group['lr'] = new_group['lr']
                    old_group['initial_lr'] = new_group['initial_lr']
                else:
                    # the group state was not saved, initialize with an empty group state
                    new_group = new_param_groups[new_group_index]
                    new_group['params'][:] = range(state_index, state_index + len(new_group['params']))
                    state_index += len(new_group['params'])
                    param_groups.append(new_group)

            state_dict['state'] = state
            state_dict['param_groups'] = param_groups

        optimizer.load_state_dict(state_dict)

    return optimizer


def create_ema(
        parameters: Iterable[Parameter] | list[dict],
        state_dict: dict | None,
        config: TrainConfig,
) -> EMAModuleWrapper | None:
    if config.ema == EMAMode.GPU:
        device = torch.device(config.train_device)
    elif config.ema == EMAMode.CPU:
        device = torch.device("cpu")
    else:
        return None

    ema = EMAModuleWrapper(
        parameters=parameters,
        decay=config.ema_decay,
        update_step_interval=config.ema_update_step_interval,
        device=device,
    )

    if state_dict is not None:
        ema.load_state_dict(state_dict)

    return ema


def create_lr_scheduler(
        config: TrainConfig,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: LearningRateScheduler,
        warmup_steps: int | float,
        num_cycles: float,
        min_factor: float,
        num_epochs: int,
        batch_size: int,
        approximate_epoch_length: int,
        gradient_accumulation_steps: int,
        global_step: int = 0,
) -> LRScheduler:
    steps_per_epoch = approximate_epoch_length
    total_steps = int(steps_per_epoch * num_epochs / gradient_accumulation_steps)

    if warmup_steps > 1:   #values > 1 are literal step count
        warmup_steps = int(warmup_steps / gradient_accumulation_steps)
    elif 0 < warmup_steps <= 1:  #values between 0-1 are treated as percentage
        warmup_steps = int(warmup_steps * total_steps)
    else:   #catch any invalid inputs or negative values
        warmup_steps = 0

    scheduler_steps = total_steps - warmup_steps

    # Force schedule-free algorithms to constant schedule.
    if config.optimizer.optimizer.is_schedule_free:
        learning_rate_scheduler = LearningRateScheduler.CONSTANT

    match learning_rate_scheduler:
        case LearningRateScheduler.CONSTANT:
            lr_lambda = lr_lambda_constant()

        case LearningRateScheduler.LINEAR:
            lr_lambda = lr_lambda_linear(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.COSINE:
            lr_lambda = lr_lambda_cosine(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.COSINE_WITH_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_restarts(
                scheduler_steps, num_cycles, min_factor
            )

        case LearningRateScheduler.COSINE_WITH_HARD_RESTARTS:
            lr_lambda = lr_lambda_cosine_with_hard_restarts(
                scheduler_steps, num_cycles, min_factor
            )

        case LearningRateScheduler.REX:
            lr_lambda = lr_lambda_rex(
                scheduler_steps, min_factor
            )

        case LearningRateScheduler.ADAFACTOR:
            from transformers.optimization import AdafactorSchedule
            return AdafactorSchedule(
                optimizer,
                initial_lr=optimizer.state_dict()['param_groups'][0]['initial_lr'],
            )
        case LearningRateScheduler.CUSTOM:
            # Special case. Unlike the others, we return from here.
            if not config.custom_learning_rate_scheduler:
                raise AssertionError("Must specify a class when using a custom LR scheduler.")
            if "." not in config.custom_learning_rate_scheduler:
                raise AssertionError("Custom class name must be in the format <module>.<class>")
            klass = config.custom_learning_rate_scheduler.split(".")[-1]
            module = config.custom_learning_rate_scheduler.removesuffix("." + klass)
            module = importlib.import_module(module)
            klass = getattr(module, klass)
            # Compile arguments into single dict.
            args = {}
            for pd in config.scheduler_params:
                key = pd["key"]
                value = pd["value"]
                # Special values
                match value:
                    case "%LR%":
                        value = config.learning_rate
                    case "%EPOCHS%":
                        value = num_epochs
                    case "%STEPS_PER_EPOCH%":
                        value = steps_per_epoch
                    case "%TOTAL_STEPS%":
                        value = total_steps
                    case "%SCHEDULER_STEPS%":
                        value = scheduler_steps
                    case _:
                        value = ast.literal_eval(value)
                args[key] = value
            scheduler = klass(optimizer=optimizer,
                              last_epoch=int(global_step / gradient_accumulation_steps) - 1,
                              **args)
            if warmup_steps > 0:
                warmup_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lr_lambda_warmup(warmup_steps, lr_lambda_constant()),
                    last_epoch=int(global_step / gradient_accumulation_steps) - 1)
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_steps],
                    last_epoch=int(global_step / gradient_accumulation_steps) - 1)
            return scheduler
        case _:
            lr_lambda = lr_lambda_constant()

    if warmup_steps > 0 and not config.optimizer.optimizer.is_schedule_free:
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

def create_trainer(
        config: TrainConfig,
        callbacks: TrainCallbacks,
        commands: TrainCommands,
        reattach: bool = False,
):
    if config.cloud.enabled:
        from modules.trainer.CloudTrainer import CloudTrainer
        trainer = CloudTrainer(config, callbacks, commands, reattach=reattach)
    elif config.multi_gpu:
        from modules.trainer.MultiTrainer import MultiTrainer
        trainer = MultiTrainer(config, callbacks, commands)
    else:
        ZLUDA.initialize_devices(config)
        from modules.trainer.GenericTrainer import GenericTrainer
        trainer = GenericTrainer(config, callbacks, commands)
    return trainer
