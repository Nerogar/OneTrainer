from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoModelSaver import HunyuanVideoModelSaver
from modules.util.enum.ModelType import ModelType

HunyuanVideoFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.HUNYUAN_VIDEO,
    model_class=HunyuanVideoModel,
    model_saver_class=HunyuanVideoModelSaver,
    embedding_saver_class=HunyuanVideoEmbeddingSaver,
)
