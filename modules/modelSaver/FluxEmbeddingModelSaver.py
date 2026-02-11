from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.util.enum.ModelType import ModelType

FluxEmbeddingModelSaver = make_embedding_model_saver(
    [ModelType.FLUX_DEV_1, ModelType.FLUX_FILL_DEV_1],
    model_class=FluxModel,
    embedding_saver_class=FluxEmbeddingSaver,
)
