from mgds.GenericDataLoaderModules import *

from modules.dataLoader.MgdsStableDiffusionXLBaseDataLoader import MgdsStablDiffusionXLBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsStableDiffusionXLEmbeddingDataLoader(MgdsStablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(MgdsStableDiffusionXLEmbeddingDataLoader, self).__init__(args, model, train_progress)

    def _load_input_modules(self, args: TrainArgs, model: StableDiffusionModel) -> list:
        modules = super(MgdsStableDiffusionXLEmbeddingDataLoader, self)._load_input_modules(args, model)

        tokens = [f"<embedding_{i}>" for i in range(model.embeddings[0].token_count)]
        all_token_string = ''.join(tokens)

        replace_text = ReplaceText(
            text_in_name='prompt', text_out_name='prompt', old_text='<embedding>', new_text=all_token_string
        )

        modules.append(replace_text)

        return modules
