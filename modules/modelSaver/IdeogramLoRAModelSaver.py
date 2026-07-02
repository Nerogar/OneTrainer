from modules.model.IdeogramModel import IdeogramModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.ideogram.IdeogramLoRASaver import IdeogramLoRASaver
from modules.util.enum.ModelType import ModelType

IdeogramLoRAModelSaver = make_lora_model_saver(
    ModelType.IDEOGRAM_4,
    model_class=IdeogramModel,
    lora_saver_class=IdeogramLoRASaver,
    embedding_saver_class=None,
)
