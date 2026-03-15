from modules.model.BaseModel import BaseModel
from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.convert.lora.convert_hunyuan_video_lora import convert_hunyuan_video_lora_key_sets
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.ModelNames import ModelNames


class HunyuanVideoLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return convert_hunyuan_video_lora_key_sets()

    def _preprocess_state_dict(self, state_dict: dict) -> dict:
        if not any(k.startswith("transformer.") for k in state_dict):
            return state_dict

        result = {}
        for k, v in state_dict.items():
            if k.startswith("transformer."):
                result["lora_transformer." + k[len("transformer."):]] = v
            elif k.startswith("lora_te1_"):
                result["lora_te2_" + k[len("lora_te1_"):]] = v
            elif k.startswith("lora_llama_"):
                result["lora_te1_" + k[len("lora_llama_"):]] = v
            else:
                result[k] = v
        return result

    def load(
            self,
            model: HunyuanVideoModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
