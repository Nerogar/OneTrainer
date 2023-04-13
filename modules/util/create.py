import torch
from torch.optim import AdamW

from modules.dataLoader.MgdsStableDiffusionDataLoader import MgdsStableDiffusionDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.FineTuneModelLoader import FineTuneModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.StableDiffusionModelSampler import StableDiffusionModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.FineTuneModelSaver import FineTuneModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.StableDiffusionFineTuneSetup import StableDiffusionFineTuneSetup
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def create_model_loader(
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            return FineTuneModelLoader()


def create_model_saver(
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver:
    match training_method:
        case TrainingMethod.FINE_TUNE:
            return FineTuneModelSaver()


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


def create_model_sampler(
        model: BaseModel,
        model_type: ModelType,
        train_device: torch.device,
) -> BaseModelSampler:
    if model_type.is_stable_diffusion():
        return StableDiffusionModelSampler(model, model_type, train_device)


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
                return MgdsStableDiffusionDataLoader(args, model, train_progress)


def create_optimizer(
        model: BaseModel,
        args: TrainArgs = None,
):
    optimizer = AdamW(
        params=model.parameters(args),
        lr=3e-6,
        weight_decay=1e-2,
        eps=1e-8,
    )

    return optimizer
