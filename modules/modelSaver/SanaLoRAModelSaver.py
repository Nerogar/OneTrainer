from modules.model.SanaModel import SanaModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.sana.SanaEmbeddingSaver import SanaEmbeddingSaver
from modules.modelSaver.sana.SanaLoRASaver import SanaLoRASaver

SanaLoRAModelSaver = make_lora_model_saver(
    model_class=SanaModel,
    lora_saver_class=SanaLoRASaver,
    embedding_saver_class=SanaEmbeddingSaver,
)
