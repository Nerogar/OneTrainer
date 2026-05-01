from modules.model.ErnieModel import ErnieModel
from modules.modelSaver.ernie.ErnieModelSaver import ErnieModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

ErnieFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.ERNIE,
    model_class=ErnieModel,
    model_saver_class=ErnieModelSaver,
    embedding_saver_class=None,
)
