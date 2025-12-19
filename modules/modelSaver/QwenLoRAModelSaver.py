from modules.model.QwenModel import QwenModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.qwen.QwenLoRASaver import QwenLoRASaver
from modules.util.enum.ModelType import ModelType

QwenLoRAModelSaver = make_lora_model_saver(
    ModelType.QWEN,
    model_class=QwenModel,
    lora_saver_class=QwenLoRASaver,
    embedding_saver_class=None,
)
