from modules.model.ChromaModel import ChromaModel
from modules.modelLoader.chroma.ChromaEmbeddingLoader import ChromaEmbeddingLoader
from modules.modelLoader.chroma.ChromaModelLoader import ChromaModelLoader
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.util.enum.ModelType import ModelType

ChromaEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma-embedding.json"},
    model_class=ChromaModel,
    model_loader_class=ChromaModelLoader,
    embedding_loader_class=ChromaEmbeddingLoader,
)
