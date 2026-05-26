from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.hunyuanVideo.HunyuanVideoEmbeddingSaver import HunyuanVideoEmbeddingSaver
from modules.modelSaver.hunyuanVideo.HunyuanVideoLoRASaver import HunyuanVideoLoRASaver
from modules.util.enum.ModelType import ModelType

HunyuanVideoLoRAModelSaver = make_lora_model_saver(
    ModelType.HUNYUAN_VIDEO,
    model_class=HunyuanVideoModel,
    lora_saver_class=HunyuanVideoLoRASaver,
    embedding_saver_class=HunyuanVideoEmbeddingSaver,
)
