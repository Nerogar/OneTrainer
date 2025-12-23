from modules.model.FluxModel import FluxModel
from modules.modelSaver.flux.FluxEmbeddingSaver import FluxEmbeddingSaver
from modules.modelSaver.flux.FluxModelSaver import FluxModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver

FluxFineTuneModelSaver = make_fine_tune_model_saver(
    model_class=FluxModel,
    model_saver_class=FluxModelSaver,
    embedding_saver_class=FluxEmbeddingSaver,
)
