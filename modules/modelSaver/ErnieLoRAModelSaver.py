from modules.model.ErnieModel import ErnieModel
from modules.modelSaver.ernie.ErnieLoRASaver import ErnieLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

ErnieLoRAModelSaver = make_lora_model_saver(
    ModelType.ERNIE,
    model_class=ErnieModel,
    lora_saver_class=ErnieLoRASaver,
    embedding_saver_class=None,
)
