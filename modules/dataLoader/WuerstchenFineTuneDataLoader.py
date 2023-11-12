from modules.dataLoader.WuerstchenBaseDataLoader import WuerstchenBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class WuerstchenFineTuneDataLoader(WuerstchenBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenFineTuneDataLoader, self).__init__(args, model, train_progress)
