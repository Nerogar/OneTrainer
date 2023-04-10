from abc import ABCMeta

import torch.nn.functional as F
from torch import Tensor

from modules.dataLoader.MgdsStableDiffusionDataLoader import MgdsStableDiffusionDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import create
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args: TrainArgs):
        self.args = args

    @staticmethod
    def __masked_mse_loss(predicted, target, mask, reduction="none"):
        masked_predicted = predicted * mask
        masked_target = target * mask
        return F.mse_loss(masked_predicted, masked_target, reduction=reduction)

    def loss(self, batch: dict, predicted: Tensor, target: Tensor) -> Tensor:
        losses = None
        match self.args.loss_function:
            case LossFunction.MSE:
                losses = F.mse_loss(predicted, target, reduction='none').mean([1, 2, 3])
            case LossFunction.MASKED_MSE:
                losses = self.__masked_mse_loss(predicted, target, mask=batch['latent_mask'], reduction='none').mean([1, 2, 3])
                # TODO: only apply if normalize masked area loss is enabled
                losses = losses * batch['latent_mask'].mean(dim=(1, 2, 3))

        return losses.mean()

    def create_model_loader(self) -> BaseModelLoader:
        return create.create_model_loader(self.args.training_method)

    def create_model_setup(self) -> BaseModelSetup:
        return create.create_model_setup(self.args.model_type, self.args.train_device, self.args.temp_device, self.args.training_method)

    def create_data_loader(self, model: BaseModel):
        model_setup = None
        match self.args.training_method:
            case TrainingMethod.FINE_TUNE:
                match self.args.model_type:
                    case ModelType.STABLE_DIFFUSION_15 \
                         | ModelType.STABLE_DIFFUSION_15_INPAINTING \
                         | ModelType.STABLE_DIFFUSION_20_DEPTH \
                         | ModelType.STABLE_DIFFUSION_20 \
                         | ModelType.STABLE_DIFFUSION_20_INPAINTING:
                        model_setup = MgdsStableDiffusionDataLoader(self.args, model)

        return model_setup

    def create_model_saver(self) -> BaseModelSaver:
        return create.create_model_saver(self.args.training_method)

    def create_model_sampler(self, model: BaseModel) -> BaseModelSampler:
        return create.create_model_sampler(model, self.args.model_type, self.args.train_device)
