from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenLoRASaver import WuerstchenLoRASaver
from modules.util.enum.ModelType import ModelType

WuerstchenLoRAModelSaver = make_lora_model_saver(
    [ModelType.WUERSTCHEN_2, ModelType.STABLE_CASCADE_1],
    model_class=WuerstchenModel,
    lora_saver_class=WuerstchenLoRASaver,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
