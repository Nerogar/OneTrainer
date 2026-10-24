from modules.model.Flux2Model import Flux2Model
from modules.modelSaver.flux2.Flux2LoRASaver import Flux2LoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

Flux2LoRAModelSaver = make_lora_model_saver(
    ModelType.FLUX_2,
    model_class=Flux2Model,
    lora_saver_class=Flux2LoRASaver,
    embedding_saver_class=None,
)
