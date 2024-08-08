from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.stableDiffusion.StableDiffusionEmbeddingLoader import StableDiffusionEmbeddingLoader
from modules.modelLoader.stableDiffusion.StableDiffusionLoRALoader import StableDiffusionLoRALoader
from modules.modelLoader.stableDiffusion.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class StableDiffusionLoRAModelLoader(
    BaseModelLoader,
    ModelSpecModelLoaderMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/sd_model_spec/sd_1.5-lora.json"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/sd_model_spec/sd_1.5_inpainting-lora.json"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/sd_model_spec/sd_2.0-lora.json"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/sd_model_spec/sd_2.0-lora.json"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/sd_model_spec/sd_2.0_inpainting-lora.json"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/sd_model_spec/sd_2.0_depth-lora.json"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/sd_model_spec/sd_2.1-lora.json"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/sd_model_spec/sd_2.1-lora.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusionModel | None:
        base_model_loader = StableDiffusionModelLoader()
        lora_model_loader = StableDiffusionLoRALoader()
        embedding_loader = StableDiffusionEmbeddingLoader()

        model = StableDiffusionModel(model_type=model_type)
        self._load_internal_data(model, model_names.lora)
        model.model_spec = self._load_default_model_spec(model_type)

        if model_names.base_model is not None:
            base_model_loader.load(model, model_type, model_names, weight_dtypes)
        lora_model_loader.load(model, model_names)
        embedding_loader.load_multiple(model, model_names)

        return model
