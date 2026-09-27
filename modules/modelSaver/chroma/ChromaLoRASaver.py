from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert_lora_util import kohya_flatten
from modules.util.convert_util import convert

from torch import Tensor


class ChromaLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _convert_legacy(self, model: ChromaModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # Single-TE model: own override so the lone text encoder stays unnumbered (lora_te, not lora_te1).
        state_dict = convert(state_dict, [
            ("transformer", "lora_transformer"),
            ("text_encoder", "lora_te"),
            ("bundle_emb", "bundle_emb"),
        ], strict=True)
        return kohya_flatten(state_dict)

    def _get_state_dict(
            self,
            model: ChromaModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.text_encoder_lora is not None:
            state_dict |= model.text_encoder_lora.state_dict()
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict

        if model.additional_embeddings and model.train_config.bundle_additional_embeddings:
            for embedding in model.additional_embeddings:
                placeholder = embedding.text_encoder_embedding.placeholder

                if embedding.text_encoder_embedding.vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.t5"] = embedding.text_encoder_embedding.vector
                if embedding.text_encoder_embedding.output_vector is not None:
                    state_dict[f"bundle_emb.{placeholder}.t5_out"] = embedding.text_encoder_embedding.output_vector

        return state_dict
