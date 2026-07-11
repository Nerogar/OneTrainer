from modules.ui.SampleParamsWindowController import SampleParamsWindowController
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig


class SamplingTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def create_new_element(self) -> SampleConfig:
        return SampleConfig.default_values(self.train_config.model_type)

    def open_element_window(self, parent, sample_config, ui_state, view_cls):
        return view_cls(parent, SampleParamsWindowController(sample_config, model_type=self.train_config.model_type), ui_state)
