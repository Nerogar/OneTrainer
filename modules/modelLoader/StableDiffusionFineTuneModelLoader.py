from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelLoaderSDConfigMixin import ModelLoaderSDConfigMixin
from modules.modelLoader.stableDiffusion.StableDiffusionEmbeddingLoader import StableDiffusionEmbeddingLoader
from modules.modelLoader.stableDiffusion.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class StableDiffusionFineTuneModelLoader(
    BaseModelLoader,
    ModelLoaderModelSpecMixin,
    ModelLoaderSDConfigMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super(StableDiffusionFineTuneModelLoader, self).__init__()

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/model_config/stable_diffusion/v1-inference.yaml"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/model_config/stable_diffusion/v1-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/model_config/stable_diffusion/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/model_config/stable_diffusion/v2-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/model_config/stable_diffusion/v2-inpainting-inference.yaml"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/model_config/stable_diffusion/v2-midas-inference.yaml"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/model_config/stable_diffusion/v2-inference-v.yaml"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/model_config/stable_diffusion/v2-inference.yaml"
            case _:
                return None

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/sd_model_spec/sd_1.5.json"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/sd_model_spec/sd_1.5_inpainting.json"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/sd_model_spec/sd_2.0.json"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/sd_model_spec/sd_2.0.json"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/sd_model_spec/sd_2.0_inpainting.json"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/sd_model_spec/sd_2.0_depth.json"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/sd_model_spec/sd_2.1.json"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/sd_model_spec/sd_2.1.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusionModel | None:
        base_model_loader = StableDiffusionModelLoader()
        embedding_loader = StableDiffusionEmbeddingLoader()

        model = StableDiffusionModel(model_type=model_type)

        self._load_internal_data(model, model_names.base_model)
        model.model_spec = self._load_default_model_spec(model_type)
        model.sd_config = self._load_sd_config(model_type)
        model.sd_config_filename = self._get_sd_config_name(model_type)

        base_model_loader.load(model, model_type, model_names, weight_dtypes)
        embedding_loader.load(model, model_names)

        return model
