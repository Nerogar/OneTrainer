from modules.model.HiDreamModel import HiDreamModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.hidream.HiDreamEmbeddingSaver import HiDreamEmbeddingSaver
from modules.modelSaver.hidream.HiDreamModelSaver import HiDreamModelSaver
from modules.util.enum.ModelType import ModelType

HiDreamFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.HI_DREAM_FULL,
    model_class=HiDreamModel,
    model_saver_class=HiDreamModelSaver,
    embedding_saver_class=HiDreamEmbeddingSaver,
)
