from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.stableDiffusion3.StableDiffusion3EmbeddingLoader import StableDiffusion3EmbeddingLoader
from modules.modelLoader.stableDiffusion3.StableDiffusion3LoRALoader import StableDiffusion3LoRALoader
from modules.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import StableDiffusion3ModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class StableDiffusion3LoRAModelLoader(
    BaseModelLoader,
    ModelSpecModelLoaderMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super(StableDiffusion3LoRAModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_3:
                return "resources/sd_model_spec/sd_3_2b_1.0-lora.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusion3Model | None:
        base_model_loader = StableDiffusion3ModelLoader()
        lora_model_loader = StableDiffusion3LoRALoader()
        embedding_loader = StableDiffusion3EmbeddingLoader()

        model = StableDiffusion3Model(model_type=model_type)
        self._load_internal_data(model, model_names.lora)
        model.model_spec = self._load_default_model_spec(model_type)

        if model_names.base_model is not None:
            base_model_loader.load(model, model_type, model_names, weight_dtypes)
        lora_model_loader.load(model, model_names)
        embedding_loader.load_multiple(model, model_names)

        return model
