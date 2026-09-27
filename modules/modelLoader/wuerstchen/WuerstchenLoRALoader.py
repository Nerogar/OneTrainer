from modules.model.WuerstchenModel import WuerstchenModel, cascade_prior_legacy
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class WuerstchenLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def _denoising_module(self, model: WuerstchenModel) -> object:
        # the canonical prior prefix is "prior", but the live module is model.prior_prior.
        return model.prior_prior

    def _legacy_conversion(self, model: WuerstchenModel) -> list | None:
        # Stable Cascade uses cascade_prior_legacy's native split-attn body; Wuerstchen v2 has no loadable
        # legacy -> None. Mirrors the saver's _convert_legacy.
        if model.model_type.is_stable_cascade():
            return [
                ("prior", "lora_prior_unet", cascade_prior_legacy),
                ("text_encoder", "lora_prior_te"),
                ("bundle_emb", "bundle_emb"),
            ]
        return None

    def load(
            self,
            model: WuerstchenModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
