from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.anima.AnimaLoRASaver import AnimaLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

AnimaLoRAModelSaver = make_lora_model_saver(
    ModelType.ANIMA,
    model_class=AnimaModel,
    lora_saver_class=AnimaLoRASaver,
    embedding_saver_class=None,
)
