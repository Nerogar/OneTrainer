
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.optimizer_util import change_optimizer


class TrainingTabController:
    def __init__(self, config: TrainConfig):
        self.config = config

    def restore_optimizer_config(self, ui_state):
        optimizer_config = change_optimizer(self.config)
        ui_state.get_var("optimizer").update(optimizer_config)

    def get_attention_mechanisms(self) -> list[tuple[str, AttentionMechanism]]:
        return [
            ("torch SDPA", AttentionMechanism.SDP),
            ("flash-attn", AttentionMechanism.FLASH),
            ("torch cuDNN", AttentionMechanism.CUDNN),
            ("torch FlexAttention", AttentionMechanism.FLEX),
        ]

    def get_layer_presets(self) -> dict:
        cls = create.get_model_setup_class(self.config.model_type, self.config.training_method)
        return cls.LAYER_PRESETS if cls is not None else {"full": []}

    def is_flow_matching(self) -> bool:
        return self.config.model_type.is_flow_matching()

    def is_custom_scheduler_value(self, value: str) -> bool:
        return value == "CUSTOM"
