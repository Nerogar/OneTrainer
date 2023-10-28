import torch

from modules.dataLoader.KandinskyBaseDataLoader import KandinskyBaseDataLoader
from modules.model.KandinskyModel import KandinskyModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class KandinskyFineTuneDataLoader(KandinskyBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: KandinskyModel,
            train_progress: TrainProgress,
    ):
        super(KandinskyFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            args,
            model,
            train_progress,
        )



