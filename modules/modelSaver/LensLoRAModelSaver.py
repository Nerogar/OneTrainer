from modules.model.LensModel import LensModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.lens.LensLoRASaver import LensLoRASaver
from modules.util.enum.ModelType import ModelType

LensLoRAModelSaver = make_lora_model_saver(
    ModelType.LENS,
    model_class=LensModel,
    lora_saver_class=LensLoRASaver,
    embedding_saver_class=None,
)
