from modules.model.ZImageModel import ZImageModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.zImage.ZImageLoRASaver import ZImageLoRASaver

ZImageLoRAModelSaver = make_lora_model_saver(
    model_class=ZImageModel,
    lora_saver_class=ZImageLoRASaver,
    embedding_saver_class=None,
)
