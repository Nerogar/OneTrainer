from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class StableDiffusionLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    # KOHYA/LEGACY for SD keep diffusers UNet names (sd-scripts wraps the diffusers UNet for non-SDXL, never
    # sgm), and the two formats are byte-identical -> both detect as "kohya" and reverse with no conversion.
    # This is why SD has no kohya-vs-legacy ambiguity (unlike SDXL, where kohya is sgm). That divergence now
    # lives in StableDiffusionModel.lora_diffusers_to_kohya() (returns None), so no loader override is needed.

    def load(
            self,
            model: StableDiffusionModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
