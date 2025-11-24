from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.chroma.ChromaEmbeddingLoader import ChromaEmbeddingLoader
from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.util.enum.ModelType import ModelType

ChromaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma.json"},
    model_class=ChromaModel,
    model_loader_class=ChromaModelLoader,
    embedding_loader_class=ChromaEmbeddingLoader,
)
