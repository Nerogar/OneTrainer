from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLEmbeddingLoader import StableDiffusionXLEmbeddingLoader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModuleFilter import ModuleFilter


class StableDiffusionXLFineTuneModelLoader(
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
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/sd_model_spec/sd_xl_base_1.0.json"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
                return "resources/sd_model_spec/sd_xl_base_1.0_inpainting.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quant_filters: list[ModuleFilter] | None = None,
    ) -> StableDiffusionXLModel | None:
        base_model_loader = StableDiffusionXLModelLoader()
        embedding_loader = StableDiffusionXLEmbeddingLoader()

        model = StableDiffusionXLModel(model_type=model_type)

        self._load_internal_data(model, model_names.base_model)
        model.model_spec = self._load_default_model_spec(model_type)

        base_model_loader.load(model, model_type, model_names, weight_dtypes, quant_filters)
        embedding_loader.load(model, model_names.base_model, model_names)

        return model
