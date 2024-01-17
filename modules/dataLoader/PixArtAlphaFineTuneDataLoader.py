import torch

from modules.dataLoader.PixArtAlphaBaseDataLoader import PixArtAlphaBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class PixArtAlphaFineTuneDataLoader(PixArtAlphaBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(PixArtAlphaFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            args,
            model,
            train_progress,
        )



