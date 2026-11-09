from modules.model.MossModel import MossModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.moss.MossModelSaver import MossModelSaver
from modules.util.enum.ModelType import ModelType

MossFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.MOSS,
    model_class=MossModel,
    model_saver_class=MossModelSaver,
    embedding_saver_class=None,
)
