from modules.model.SanaModel import SanaModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.sana.SanaEmbeddingLoader import SanaEmbeddingLoader
from modules.modelLoader.sana.SanaModelLoader import SanaModelLoader
from modules.util.enum.ModelType import ModelType

SanaEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.SANA: "resources/sd_model_spec/sana-embedding.json"},
    model_class=SanaModel,
    model_loader_class=SanaModelLoader,
    embedding_loader_class=SanaEmbeddingLoader,
)
