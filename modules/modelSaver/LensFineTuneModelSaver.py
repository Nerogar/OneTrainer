from modules.model.LensModel import LensModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.lens.LensModelSaver import LensModelSaver
from modules.util.enum.ModelType import ModelType

LensFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.LENS,
    model_class=LensModel,
    model_saver_class=LensModelSaver,
    embedding_saver_class=None,
)
