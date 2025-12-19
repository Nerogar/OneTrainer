from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.wuerstchen.WuerstchenEmbeddingSaver import WuerstchenEmbeddingSaver
from modules.modelSaver.wuerstchen.WuerstchenModelSaver import WuerstchenModelSaver
from modules.util.enum.ModelType import ModelType

WuerstchenFineTuneModelSaver = make_fine_tune_model_saver(
    [ModelType.WUERSTCHEN_2, ModelType.STABLE_CASCADE_1],
    model_class=WuerstchenModel,
    model_saver_class=WuerstchenModelSaver,
    embedding_saver_class=WuerstchenEmbeddingSaver,
)
