from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenLoRASaver import WuerstchenLoRASaver

WuerstchenLoRAModelSaver = make_lora_model_saver(
    model_class=WuerstchenModel,
    lora_saver_class=WuerstchenLoRASaver,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
