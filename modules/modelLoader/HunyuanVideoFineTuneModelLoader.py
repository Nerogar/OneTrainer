from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.hunyuanVideo.HunyuanVideoEmbeddingLoader import HunyuanVideoEmbeddingLoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoModelLoader import HunyuanVideoModelLoader
from modules.util.enum.ModelType import ModelType

HunyuanVideoFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={ModelType.HUNYUAN_VIDEO: "resources/sd_model_spec/hunyuan_video.json"},
    model_class=HunyuanVideoModel,
    model_loader_class=HunyuanVideoModelLoader,
    embedding_loader_class=HunyuanVideoEmbeddingLoader,
)
