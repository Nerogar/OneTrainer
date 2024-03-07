import torch
from mgds.pipelineModules.ReplaceText import ReplaceText

from modules.dataLoader.WuerstchenBaseDataLoader import WuerstchenBaseDataLoader
from modules.model.WuerstchenModel import WuerstchenModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class WuerstchenEmbeddingDataLoader(WuerstchenBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: WuerstchenModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenEmbeddingDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )

    def _load_input_modules(self, config: TrainConfig, model: WuerstchenModel) -> list:
        modules = super(WuerstchenEmbeddingDataLoader, self)._load_input_modules(config, model)

        all_token_string = ''.join(model.embeddings[0].text_tokens)

        replace_text = ReplaceText(
            text_in_name='prompt', text_out_name='prompt', old_text='<embedding>', new_text=all_token_string
        )

        modules.append(replace_text)

        return modules
