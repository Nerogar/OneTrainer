import torch

from modules.dataLoader.StableDiffusionBaseDataLoader import StablDiffusionBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionFineTuneDataLoader(StablDiffusionBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )



