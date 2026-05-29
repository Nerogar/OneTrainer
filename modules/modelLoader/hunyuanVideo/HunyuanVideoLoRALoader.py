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
                # Reverse ComfyUI conditioning MLPEmbedder renames → OT OMI paths
                k = k.replace(".guidance_in.in_layer.", ".guidance_in.mlp.0.")
                k = k.replace(".guidance_in.out_layer.", ".guidance_in.mlp.2.")
                k = k.replace(".time_in.in_layer.", ".time_in.mlp.0.")
                k = k.replace(".time_in.out_layer.", ".time_in.mlp.2.")
                k = k.replace(".txt_in.c_embedder.in_layer.", ".txt_in.c_embedder.linear_1.")
                k = k.replace(".txt_in.c_embedder.out_layer.", ".txt_in.c_embedder.linear_2.")
                k = k.replace(".txt_in.t_embedder.in_layer.", ".txt_in.t_embedder.linear_1.")
                k = k.replace(".txt_in.t_embedder.out_layer.", ".txt_in.t_embedder.linear_2.")
                # Reverse fc1/fc2 → mlp.0/mlp.2 for transformer block MLPs
                k = k.replace(".mlp.fc1.", ".mlp.0.")
                k = k.replace(".mlp.fc2.", ".mlp.2.")
                k = k.replace(".fc1.", ".fc0.")
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
