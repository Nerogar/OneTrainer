from modules.model.MossModel import MossModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.moss.MossEmbeddingLoader import MossEmbeddingLoader
from modules.modelLoader.moss.MossModelLoader import MossModelLoader
from modules.util.enum.ModelType import ModelType

MossEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.MOSS: "resources/sd_model_spec/moss-embedding.json"},
    model_class=MossModel,
    model_loader_class=MossModelLoader,
    embedding_loader_class=MossEmbeddingLoader,
)
