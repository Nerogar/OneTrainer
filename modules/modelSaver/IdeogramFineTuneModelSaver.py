from modules.model.IdeogramModel import IdeogramModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.ideogram.IdeogramModelSaver import IdeogramModelSaver
from modules.util.enum.ModelType import ModelType

IdeogramFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.IDEOGRAM_4,
    model_class=IdeogramModel,
    model_saver_class=IdeogramModelSaver,
    embedding_saver_class=None,
)
