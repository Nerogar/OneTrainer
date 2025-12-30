from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver

HunyuanVideoEmbeddingModelSaver = make_embedding_model_saver(
    model_class=HunyuanVideoModel,
    embedding_saver_class=HunyuanVideoEmbeddingSaver,
)
