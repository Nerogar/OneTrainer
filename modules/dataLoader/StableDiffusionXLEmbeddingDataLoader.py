from mgds.GenericDataLoaderModules import *

from modules.dataLoader.StableDiffusionXLBaseDataLoader import StablDiffusionXLBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class StableDiffusionXLEmbeddingDataLoader(StablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionXLEmbeddingDataLoader, self).__init__(
            train_device,
            temp_device,
            args,
            model,
            train_progress,
        )

    def _load_input_modules(self, args: TrainArgs, model: StableDiffusionXLModel) -> list:
        modules = super(StableDiffusionXLEmbeddingDataLoader, self)._load_input_modules(args, model)

        all_token_string = ''.join(model.embeddings[0].text_tokens)

        replace_text = ReplaceText(
            text_in_name='prompt', text_out_name='prompt', old_text='<embedding>', new_text=all_token_string
        )

        modules.append(replace_text)

        return modules
