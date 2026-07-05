from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class PixArtAlphaLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def _legacy_conversion(self, model: PixArtAlphaModel) -> list | None:
        # Single-TE override: no-digit lora_te (the default numbers it lora_te1); mirrors the saver's _convert_legacy.
        return [
            ("transformer", "lora_transformer"),
            ("text_encoder", "lora_te"),
            ("bundle_emb", "bundle_emb"),
        ]

    def load(
            self,
            model: PixArtAlphaModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
