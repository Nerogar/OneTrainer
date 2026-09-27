from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler


class SchedulerParamsWindowController:
    def __init__(self, config: TrainConfig):
        self.config = config

    def is_custom_scheduler(self) -> bool:
        return self.config.learning_rate_scheduler is LearningRateScheduler.CUSTOM

class KvParamsController:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def create_new_element(self) -> dict[str, str]:
        return {"key": "", "value": ""}
