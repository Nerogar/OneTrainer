import ast
import importlib
from typing import Iterable

import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler, UniPCMultistepScheduler, SchedulerMixin
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR

from modules.dataLoader.PixArtAlphaBaseDataLoader import PixArtAlphaBaseDataLoader
from modules.dataLoader.StableDiffusionBaseDataLoader import StableDiffusionBaseDataLoader
from modules.dataLoader.StableDiffusionFineTuneVaeDataLoader import StableDiffusionFineTuneVaeDataLoader
from modules.dataLoader.StableDiffusionXLBaseDataLoader import StableDiffusionXLBaseDataLoader
from modules.dataLoader.WuerstchenBaseDataLoader import WuerstchenBaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.PixArtAlphaEmbeddingModelLoader import PixArtAlphaEmbeddingModelLoader
from modules.modelLoader.PixArtAlphaFineTuneModelLoader import PixArtAlphaFineTuneModelLoader
from modules.modelLoader.PixArtAlphaLoRAModelLoader import PixArtAlphaLoRAModelLoader
from modules.modelLoader.StableDiffusionEmbeddingModelLoader import StableDiffusionEmbeddingModelLoader
from modules.modelLoader.StableDiffusionFineTuneModelLoader import StableDiffusionFineTuneModelLoader
from modules.modelLoader.StableDiffusionLoRAModelLoader import StableDiffusionLoRAModelLoader
from modules.modelLoader.StableDiffusionXLEmbeddingModelLoader import StableDiffusionXLEmbeddingModelLoader
from modules.modelLoader.StableDiffusionXLFineTuneModelLoader import StableDiffusionXLFineTuneModelLoader
from modules.modelLoader.StableDiffusionXLLoRAModelLoader import StableDiffusionXLLoRAModelLoader
from modules.modelLoader.WuerstchenEmbeddingModelLoader import WuerstchenEmbeddingModelLoader
from modules.modelLoader.WuerstchenFineTuneModelLoader import WuerstchenFineTuneModelLoader
from modules.modelLoader.WuerstchenLoRAModelLoader import WuerstchenLoRAModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.PixArtAlphaSampler import PixArtAlphaSampler
from modules.modelSampler.StableDiffusionSampler import StableDiffusionSampler
from modules.modelSampler.StableDiffusionVaeSampler import StableDiffusionVaeSampler
from modules.modelSampler.StableDiffusionXLSampler import StableDiffusionXLSampler
from modules.modelSampler.WuerstchenSampler import WuerstchenSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.PixArtAlphaEmbeddingModelSaver import PixArtAlphaEmbeddingModelSaver
from modules.modelSaver.PixArtAlphaFineTuneModelSaver import PixArtAlphaFineTuneModelSaver
from modules.modelSaver.PixArtAlphaLoRAModelSaver import PixArtAlphaLoRAModelSaver
from modules.modelSaver.StableDiffusionEmbeddingModelSaver import StableDiffusionEmbeddingModelSaver
from modules.modelSaver.StableDiffusionFineTuneModelSaver import StableDiffusionFineTuneModelSaver
from modules.modelSaver.StableDiffusionLoRAModelSaver import StableDiffusionLoRAModelSaver
from modules.modelSaver.StableDiffusionXLEmbeddingModelSaver import StableDiffusionXLEmbeddingModelSaver
from modules.modelSaver.StableDiffusionXLFineTuneModelSaver import StableDiffusionXLFineTuneModelSaver
from modules.modelSaver.StableDiffusionXLLoRAModelSaver import StableDiffusionXLLoRAModelSaver
from modules.modelSaver.WuerstchenEmbeddingModelSaver import WuerstchenEmbeddingModelSaver
from modules.modelSaver.WuerstchenFineTuneModelSaver import WuerstchenFineTuneModelSaver
from modules.modelSaver.WuerstchenLoRAModelSaver import WuerstchenLoRAModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.PixArtAlphaEmbeddingSetup import PixArtAlphaEmbeddingSetup
from modules.modelSetup.PixArtAlphaFineTuneSetup import PixArtAlphaFineTuneSetup
from modules.modelSetup.PixArtAlphaLoRASetup import PixArtAlphaLoRASetup
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
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.lr_scheduler_util import *
from modules.util.optimizer.CAME import CAME
from modules.util.optimizer.adafactor_extensions import patch_adafactor
from modules.util.optimizer.adam_extensions import patch_adam
from modules.util.optimizer.adamw_extensions import patch_adamw


def create_model_loader(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLFineTuneModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenFineTuneModelLoader()
            if model_type.is_pixart():
                return PixArtAlphaFineTuneModelLoader()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneModelLoader()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenLoRAModelLoader()
            if model_type.is_pixart():
                return PixArtAlphaLoRAModelLoader()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelLoader()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelLoader()
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingModelLoader()
            if model_type.is_pixart():
                return PixArtAlphaEmbeddingModelLoader()


def create_model_saver(
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLFineTuneModelSaver()
            if model_type.is_wuerstchen():
                return WuerstchenFineTuneModelSaver()
            if model_type.is_pixart():
                return PixArtAlphaFineTuneModelSaver()
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneModelSaver()
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionLoRAModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLLoRAModelSaver()
            if model_type.is_wuerstchen():
                return WuerstchenLoRAModelSaver()
            if model_type.is_pixart():
                return PixArtAlphaLoRAModelSaver()
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingModelSaver()
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingModelSaver()
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingModelSaver()
            if model_type.is_pixart():
                return PixArtAlphaEmbeddingModelSaver()


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
            if model_type.is_pixart():
                return PixArtAlphaFineTuneSetup(train_device, temp_device, debug_mode)
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
            if model_type.is_pixart():
                return PixArtAlphaLoRASetup(train_device, temp_device, debug_mode)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_wuerstchen():
                return WuerstchenEmbeddingSetup(train_device, temp_device, debug_mode)
            if model_type.is_pixart():
                return PixArtAlphaEmbeddingSetup(train_device, temp_device, debug_mode)


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
            if model_type.is_pixart():
                return PixArtAlphaSampler(train_device, temp_device, model, model_type)
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
            if model_type.is_pixart():
                return PixArtAlphaSampler(train_device, temp_device, model, model_type)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionSampler(train_device, temp_device, model, model_type)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLSampler(train_device, temp_device, model, model_type)
            if model_type.is_wuerstchen():
                return WuerstchenSampler(train_device, temp_device, model, model_type)
            if model_type.is_pixart():
                return PixArtAlphaSampler(train_device, temp_device, model, model_type)


def create_data_loader(
        train_device: torch.device,
        temp_device: torch.device,
        model: BaseModel,
        model_type: ModelType,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        config: TrainConfig = None,
        train_progress: TrainProgress = TrainProgress(),
):
    match training_method:
        case TrainingMethod.FINE_TUNE:
            if model_type.is_stable_diffusion():
                return StableDiffusionBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_pixart():
                return PixArtAlphaBaseDataLoader(train_device, temp_device, config, model, train_progress)
        case TrainingMethod.FINE_TUNE_VAE:
            if model_type.is_stable_diffusion():
                return StableDiffusionFineTuneVaeDataLoader(train_device, temp_device, config, model, train_progress)
        case TrainingMethod.LORA:
            if model_type.is_stable_diffusion():
                return StableDiffusionBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_pixart():
                return PixArtAlphaBaseDataLoader(train_device, temp_device, config, model, train_progress)
        case TrainingMethod.EMBEDDING:
            if model_type.is_stable_diffusion():
                return StableDiffusionBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_stable_diffusion_xl():
                return StableDiffusionXLBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_wuerstchen():
                return WuerstchenBaseDataLoader(train_device, temp_device, config, model, train_progress)
            if model_type.is_pixart():
                return PixArtAlphaBaseDataLoader(train_device, temp_device, config, model, train_progress)


def create_optimizer(
        parameter_group_collection: NamedParameterGroupCollection,
        state_dict: dict | None,
        config: TrainConfig,
) -> torch.optim.Optimizer:
    optimizer = None
    optimizer_config = config.optimizer

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
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
                amsgrad=optimizer_config.amsgrad if optimizer_config.amsgrad is not None else False,
                adam_w_mode=optimizer_config.adam_w_mode if optimizer_config.adam_w_mode is not None else True,
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
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999),
                bias_correction=optimizer_config.bias_correction if optimizer_config.bias_correction is not None else True,
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
                lr=None if optimizer_config.relative_step == True else config.learning_rate,
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
            optimizer = CAME(
                params=parameters,
                lr=config.learning_rate,
                eps=(optimizer_config.eps if optimizer_config.eps is not None else 1e-30,
                     optimizer_config.eps2 if optimizer_config.eps2 is not None else 1e-16),
                betas=(optimizer_config.beta1 if optimizer_config.beta1 is not None else 0.9,
                       optimizer_config.beta2 if optimizer_config.beta2 is not None else 0.999,
                       optimizer_config.beta3 if optimizer_config.beta3 is not None else 0.9999),
                weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0,
                stochastic_rounding=optimizer_config.stochastic_rounding
            )

    if state_dict is not None:
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
                    for i, old_state_index in enumerate(new_group['params']):
                        new_group['params'][i] = state_index
                        state_index += 1
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
        warmup_steps: int,
        num_cycles: float,
        num_epochs: int,
        batch_size: int,
        approximate_epoch_length: int,
        gradient_accumulation_steps: int,
        global_step: int = 0,
) -> LRScheduler:
    steps_per_epoch = approximate_epoch_length
    total_steps = int(steps_per_epoch * num_epochs / gradient_accumulation_steps)
    warmup_steps = int(warmup_steps / gradient_accumulation_steps)
    scheduler_steps = total_steps - warmup_steps

    # Force schedule-free algorithms to constant schedule.
    if config.optimizer.optimizer.is_schedule_free:
        learning_rate_scheduler = LearningRateScheduler.CONSTANT

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
