from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.chroma.ChromaLoRASaver import ChromaLoRASaver
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

ChromaLoRAModelSaver = make_lora_model_saver(
    ModelType.CHROMA_1,
    model_class=ChromaModel,
    lora_saver_class=ChromaLoRASaver,
    embedding_saver_class=ChromaEmbeddingSaver,
)
