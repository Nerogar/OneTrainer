from modules.model.MossModel import MossModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.moss.MossLoRASaver import MossLoRASaver
from modules.util.enum.ModelType import ModelType

MossLoRAModelSaver = make_lora_model_saver(
    ModelType.MOSS,
    model_class=MossModel,
    lora_saver_class=MossLoRASaver,
    embedding_saver_class=None,
)
