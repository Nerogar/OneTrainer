from mgds.GenericDataLoaderModules import *

from modules.dataLoader.MgdsStableDiffusionBaseDataLoader import MgdsStablDiffusionBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsStableDiffusionFineTuneDataLoader(MgdsStablDiffusionBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(MgdsStableDiffusionFineTuneDataLoader, self).__init__(args, model, train_progress)



