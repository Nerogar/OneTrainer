from modules.model.LTXModel import LTXModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.ltx2.LTXLoRASaver import LTXLoRASaver
from modules.util.enum.ModelType import ModelType

LTXLoRAModelSaver = make_lora_model_saver(
    ModelType.LTX_2,
    model_class=LTXModel,
    lora_saver_class=LTXLoRASaver,
    embedding_saver_class=None,
)
