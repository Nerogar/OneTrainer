from modules.model.ZImageModel import ZImageModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.zImage.ZImageModelSaver import ZImageModelSaver

ZImageFineTuneModelSaver = make_fine_tune_model_saver(
    model_class=ZImageModel,
    model_saver_class=ZImageModelSaver,
    embedding_saver_class=None,
)
