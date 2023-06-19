from modules.dataLoader.MgdsKandinskyBaseDataLoader import MgdsKandinskyBaseDataLoader
from modules.model.KandinskyModel import KandinskyModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsKandinskyFineTuneDataLoader(MgdsKandinskyBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: KandinskyModel,
            train_progress: TrainProgress,
    ):
        super(MgdsKandinskyFineTuneDataLoader, self).__init__(args, model, train_progress)



