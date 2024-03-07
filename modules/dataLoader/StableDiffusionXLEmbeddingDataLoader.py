import torch
from mgds.pipelineModules.ReplaceText import ReplaceText

from modules.dataLoader.StableDiffusionXLBaseDataLoader import StablDiffusionXLBaseDataLoader
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionXLEmbeddingDataLoader(StablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionXLEmbeddingDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )

    def _load_input_modules(self, config: TrainConfig, model: StableDiffusionXLModel) -> list:
        modules = super(StableDiffusionXLEmbeddingDataLoader, self)._load_input_modules(config, model)

        all_token_string = ''.join(model.embeddings[0].text_tokens)

        replace_text = ReplaceText(
            text_in_name='prompt', text_out_name='prompt', old_text='<embedding>', new_text=all_token_string
        )

        modules.append(replace_text)

        return modules
