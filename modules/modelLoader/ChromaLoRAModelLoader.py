from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.chroma.ChromaEmbeddingLoader import ChromaEmbeddingLoader
from modules.modelLoader.chroma.ChromaLoRALoader import ChromaLoRALoader
from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.util.enum.ModelType import ModelType

ChromaLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma-lora.json"},
    model_class=ChromaModel,
    model_loader_class=ChromaModelLoader,
    embedding_loader_class=ChromaEmbeddingLoader,
    lora_loader_class=ChromaLoRALoader,
)
