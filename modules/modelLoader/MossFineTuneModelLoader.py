from modules.model.MossModel import MossModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.moss.MossModelLoader import MossModelLoader
from modules.util.enum.ModelType import ModelType

MossFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.MOSS: "resources/sd_model_spec/moss.json"},
    model_class=MossModel,
    model_loader_class=MossModelLoader,
    embedding_loader_class=None,
)
