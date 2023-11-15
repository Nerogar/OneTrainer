import torch

from modules.dataLoader.WuerstchenBaseDataLoader import WuerstchenBaseDataLoader
from modules.model.WuerstchenModel import WuerstchenModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenFineTuneDataLoader(WuerstchenBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: WuerstchenModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            args,
            model,
            train_progress,
        )
