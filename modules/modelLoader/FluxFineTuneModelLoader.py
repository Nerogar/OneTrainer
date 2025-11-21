from modules.model.FluxModel import FluxModel
from modules.modelLoader.flux.FluxEmbeddingLoader import FluxEmbeddingLoader
from modules.modelLoader.flux.FluxModelLoader import FluxModelLoader
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.util.enum.ModelType import ModelType

FluxFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.FLUX_DEV_1: "resources/sd_model_spec/flux_dev_1.0.json",
        ModelType.FLUX_FILL_DEV_1: "resources/sd_model_spec/flux_dev_fill_1.0.json",
    },
    model_class=FluxModel,
    model_loader_class=FluxModelLoader,
    embedding_loader_class=FluxEmbeddingLoader,
)
