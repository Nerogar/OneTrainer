from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenModelSaver import WuerstchenModelSaver

WuerstchenFineTuneModelSaver = make_fine_tune_model_saver(
    model_class=WuerstchenModel,
    model_saver_class=WuerstchenModelSaver,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
