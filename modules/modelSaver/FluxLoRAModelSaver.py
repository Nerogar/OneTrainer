from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.flux.FluxLoRASaver import FluxLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

FluxLoRAModelSaver = make_lora_model_saver(
    [ModelType.FLUX_DEV_1, ModelType.FLUX_FILL_DEV_1],
    model_class=FluxModel,
    lora_saver_class=FluxLoRASaver,
    embedding_saver_class=FluxEmbeddingSaver,
)
