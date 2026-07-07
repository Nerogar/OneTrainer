from modules.model.WuerstchenModel import WuerstchenModel, cascade_prior_legacy
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert_lora_util import kohya_flatten
from modules.util.convert_util import convert

from torch import Tensor


class WuerstchenLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _convert_legacy(self, model: WuerstchenModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # Stable Cascade has a real, loadable legacy (flattened split-attn body). Wuerstchen v2's only
        # historical output was a verbatim-dotted dict the loader could never un-prefix, so it's dropped
        # (like Sana/HiDream).
        if model.model_type.is_stable_cascade():
            state_dict = convert(state_dict, [
                ("prior", "lora_prior_unet", cascade_prior_legacy),
                ("text_encoder", "lora_prior_te"),
                ("bundle_emb", "bundle_emb"),
            ], strict=True)
            return kohya_flatten(state_dict)
        raise NotImplementedError(
            "The LEGACY LoRA output format is not supported for Wuerstchen v2 (its only prior LEGACY output "
            "was a never-loadable dotted format). Use DIFFUSERS_LORA or KOHYA_LORA.")

    def _get_state_dict(
            self,
            model: WuerstchenModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.prior_text_encoder_lora is not None:
            state_dict |= model.prior_text_encoder_lora.state_dict()
        if model.prior_prior_lora is not None:
            state_dict |= model.prior_prior_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.prior_text_encoder_embedding.placeholder

                if embedding.prior_text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_g"] = embedding.prior_text_encoder_embedding.vector
                if embedding.prior_text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.clip_g_out"] = embedding.prior_text_encoder_embedding.output_vector

        return state_dict
