from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.krea2.Krea2LoRASaver import Krea2LoRASaver
from modules.util.enum.ModelType import ModelType

Krea2LoRAModelSaver = make_lora_model_saver(
    ModelType.KREA_2,
    model_class=Krea2Model,
    lora_saver_class=Krea2LoRASaver,
    embedding_saver_class=None,
)
