from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver

FluxEmbeddingModelSaver = make_embedding_model_saver(
    model_class=FluxModel,
    embedding_saver_class=FluxEmbeddingSaver,
)
