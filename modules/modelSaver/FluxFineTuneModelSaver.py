from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.flux.FluxModelSaver import FluxModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

FluxFineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.FLUX_DEV_1, ModelType.FLUX_FILL_DEV_1],
    model_class=FluxModel,
    model_saver_class=FluxModelSaver,
    embedding_saver_class=FluxEmbeddingSaver,
)
