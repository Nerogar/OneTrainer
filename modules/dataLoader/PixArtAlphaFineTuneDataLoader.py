import torch

from modules.dataLoader.PixArtAlphaBaseDataLoader import PixArtAlphaBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class PixArtAlphaFineTuneDataLoader(PixArtAlphaBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(PixArtAlphaFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )



