from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.modelSaver.hidream.HiDreamLoRASaver import HiDreamLoRASaver

HiDreamLoRAModelSaver = make_lora_model_saver(
    model_class=HiDreamModel,
    lora_saver_class=HiDreamLoRASaver,
    embedding_saver_class=HiDreamEmbeddingSaver,
)
