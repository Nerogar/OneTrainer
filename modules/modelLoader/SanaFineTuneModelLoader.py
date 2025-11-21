from modules.model.SanaModel import SanaModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.sana.SanaEmbeddingLoader import SanaEmbeddingLoader
from modules.modelLoader.sana.SanaModelLoader import SanaModelLoader
from modules.util.enum.ModelType import ModelType

SanaFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.SANA: "resources/sd_model_spec/sana.json"},
    model_class=SanaModel,
    model_loader_class=SanaModelLoader,
    embedding_loader_class=SanaEmbeddingLoader,
)
