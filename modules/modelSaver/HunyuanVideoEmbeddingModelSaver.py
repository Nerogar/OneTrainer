from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.util.enum.ModelType import ModelType

HunyuanVideoEmbeddingModelSaver = make_embedding_model_saver(
    ModelType.HUNYUAN_VIDEO,
    model_class=HunyuanVideoModel,
    embedding_saver_class=HunyuanVideoEmbeddingSaver,
)
