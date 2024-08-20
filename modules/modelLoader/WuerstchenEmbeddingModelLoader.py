from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.wuerstchen.WuerstchenEmbeddingLoader import WuerstchenEmbeddingLoader
from modules.modelLoader.wuerstchen.WuerstchenModelLoader import WuerstchenModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class WuerstchenEmbeddingModelLoader(
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
            case ModelType.WUERSTCHEN_2:
                return "resources/sd_model_spec/wuerstchen_2.0-embedding.json"
            case ModelType.STABLE_CASCADE_1:
                return "resources/sd_model_spec/stable_cascade_1.0-embedding.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> WuerstchenModel | None:
        base_model_loader = WuerstchenModelLoader()
        embedding_loader = WuerstchenEmbeddingLoader()

        model = WuerstchenModel(model_type=model_type)

        if model_names.base_model:
            base_model_loader.load(model, model_type, model_names, weight_dtypes)
        embedding_loader.load_multiple(model, model_names)
        embedding_loader.load_single(model, model_names)
        self._load_internal_data(model, model_names.embedding.model_name)

        model.model_spec = self._load_default_model_spec(model_type)

        return model
