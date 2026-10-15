
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig


class AdditionalEmbeddingsTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def create_new_element(self) -> TrainEmbeddingConfig:
        return TrainEmbeddingConfig.default_values()

    def randomize_uuid(self, embedding_config: TrainEmbeddingConfig) -> TrainEmbeddingConfig:
        embedding_config.uuid = TrainEmbeddingConfig.default_values().uuid
        return embedding_config
