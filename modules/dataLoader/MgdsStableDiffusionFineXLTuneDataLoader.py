from modules.dataLoader.MgdsStableDiffusionXLBaseDataLoader import MgdsStablDiffusionXLBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsStableDiffusionXLFineTuneDataLoader(MgdsStablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(MgdsStableDiffusionXLFineTuneDataLoader, self).__init__(args, model, train_progress)
