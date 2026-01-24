from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.flux.FluxLoRASaver import FluxLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver

FluxLoRAModelSaver = make_lora_model_saver(
    model_class=FluxModel,
    lora_saver_class=FluxLoRASaver,
    embedding_saver_class=FluxEmbeddingSaver,
)
