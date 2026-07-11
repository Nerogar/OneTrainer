from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.krea2.Krea2ModelSaver import Krea2ModelSaver
from modules.util.enum.ModelType import ModelType

Krea2FineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.KREA_2,
    model_class=Krea2Model,
    model_saver_class=Krea2ModelSaver,
    embedding_saver_class=None,
)
