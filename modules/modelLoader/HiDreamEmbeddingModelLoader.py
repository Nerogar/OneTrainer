from modules.model.HiDreamModel import HiDreamModel
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.hiDream.HiDreamEmbeddingLoader import HiDreamEmbeddingLoader
from modules.modelLoader.hiDream.HiDreamModelLoader import HiDreamModelLoader
from modules.util.enum.ModelType import ModelType

HiDreamEmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.HI_DREAM_FULL: "resources/sd_model_spec/hi_dream_full-embedding.json"},
    model_class=HiDreamModel,
    model_loader_class=HiDreamModelLoader,
    embedding_loader_class=HiDreamEmbeddingLoader,
)
