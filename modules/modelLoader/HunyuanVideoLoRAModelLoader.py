from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.hunyuanVideo.HunyuanVideoEmbeddingLoader import HunyuanVideoEmbeddingLoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoLoRALoader import HunyuanVideoLoRALoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoModelLoader import HunyuanVideoModelLoader
from modules.util.enum.ModelType import ModelType

HunyuanVideoLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.HUNYUAN_VIDEO: "resources/sd_model_spec/hunyuan_video-lora.json"},
    model_class=HunyuanVideoModel,
    model_loader_class=HunyuanVideoModelLoader,
    embedding_loader_class=HunyuanVideoEmbeddingLoader,
    lora_loader_class=HunyuanVideoLoRALoader,
)
