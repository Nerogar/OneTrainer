from modules.model.ZImageModel import ZImageModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.zImage.ZImageModelSaver import ZImageModelSaver
from modules.util.enum.ModelType import ModelType

ZImageFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.Z_IMAGE,
    model_class=ZImageModel,
    model_saver_class=ZImageModelSaver,
    embedding_saver_class=None,
)
