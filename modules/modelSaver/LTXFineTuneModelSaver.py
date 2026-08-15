from modules.model.LTXModel import LTXModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.ltx2.LTXModelSaver import LTXModelSaver
from modules.util.enum.ModelType import ModelType

LTXFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.LTX_2,
    model_class=LTXModel,
    model_saver_class=LTXModelSaver,
    embedding_saver_class=None,
)
