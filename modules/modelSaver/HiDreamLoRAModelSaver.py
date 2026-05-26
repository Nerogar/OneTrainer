from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.modelSaver.hidream.HiDreamLoRASaver import HiDreamLoRASaver
from modules.util.enum.ModelType import ModelType

HiDreamLoRAModelSaver = make_lora_model_saver(
    ModelType.HI_DREAM_FULL,
    model_class=HiDreamModel,
    lora_saver_class=HiDreamLoRASaver,
    embedding_saver_class=HiDreamEmbeddingSaver,
)
