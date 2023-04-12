import torch

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.FineTuneModelLoader import FineTuneModelLoader
from modules.modelSampler import BaseModelSampler
from modules.modelSampler.StableDiffusionModelSampler import StableDiffusionModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSaver.FineTuneModelSaver import FineTuneModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.StableDiffusionFineTuneSetup import StableDiffusionFineTuneSetup
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def create_model_loader(
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelLoader:
    model_loader = None
    match training_method:
        case TrainingMethod.FINE_TUNE:
            model_loader = FineTuneModelLoader()

    return model_loader


def create_model_saver(
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
) -> BaseModelSaver:
    model_saver = None
    match training_method:
        case TrainingMethod.FINE_TUNE:
            model_saver = FineTuneModelSaver()
    return model_saver


def create_model_setup(
        model_type: ModelType,
        train_device: torch.device,
        temp_device: torch.device,
        training_method: TrainingMethod = TrainingMethod.FINE_TUNE,
        debug_mode: bool = False,
) -> BaseModelSetup:
    model_setup = None
    match training_method:
        case TrainingMethod.FINE_TUNE:
            match model_type:
                case ModelType.STABLE_DIFFUSION_15 \
                     | ModelType.STABLE_DIFFUSION_15_INPAINTING \
                     | ModelType.STABLE_DIFFUSION_20_DEPTH \
                     | ModelType.STABLE_DIFFUSION_20 \
                     | ModelType.STABLE_DIFFUSION_20_INPAINTING:
                    model_setup = StableDiffusionFineTuneSetup(train_device, temp_device, debug_mode)

    return model_setup


def create_model_sampler(
        model: BaseModel,
        model_type: ModelType,
        train_device: torch.device,
) -> BaseModelSampler:
    model_sampler = None
    if model_type.is_stable_diffusion():
        return StableDiffusionModelSampler(model, model_type, train_device)
