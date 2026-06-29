from modules.model.AnimaModel import AnimaModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class AnimaLoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    # Anima is new on this branch and has no frozen pre-branch LoRA output, so _convert_legacy is not
    # overridden -- it inherits the mixin raise, and ModelType.supported_lora_formats drops LEGACY_LORA.
    #
    # ORIGINAL's net. wrapper is no longer a saver override: it lives in the denoising body
    # (AnimaModel.diffusers_to_original() renames to the netless DiT names then adds net.), so the mixin's
    # generic _save_original produces net.<DiT> after it strips the transformer. component prefix. COMFY and
    # KOHYA override the body back to the netless names (AnimaModel.lora_diffusers_to_{comfy,kohya}).

    def _get_state_dict(
            self,
            model: AnimaModel,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict
