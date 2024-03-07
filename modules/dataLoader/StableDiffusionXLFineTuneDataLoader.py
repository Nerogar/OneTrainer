import torch

from modules.dataLoader.StableDiffusionXLBaseDataLoader import StablDiffusionXLBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionXLFineTuneDataLoader(StablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionXLFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )
