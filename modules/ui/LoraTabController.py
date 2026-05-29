
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType


class LoraTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_peft_types(self) -> list[tuple[str, PeftType]]:
        return [
            ("LoRA", PeftType.LORA),
            ("LoHa", PeftType.LOHA),
            ("OFT v2", PeftType.OFT_2),
            ("LoKr", PeftType.LOKR),
        ]

    def get_lora_weight_dtypes(self) -> list[tuple[str, DataType]]:
        return [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ]
