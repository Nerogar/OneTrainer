from modules.model.Flux2Model import Flux2Model
from modules.modelSaver.flux2.Flux2ModelSaver import Flux2ModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

Flux2FineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.FLUX_2,
    model_class=Flux2Model,
    model_saver_class=Flux2ModelSaver,
    embedding_saver_class=None,
)
