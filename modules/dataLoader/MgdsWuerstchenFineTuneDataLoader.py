from modules.dataLoader.MgdsWuerstchenBaseDataLoader import MgdsWuerstchenBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsWuerstchenFineTuneDataLoader(MgdsWuerstchenBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(MgdsWuerstchenFineTuneDataLoader, self).__init__(args, model, train_progress)
