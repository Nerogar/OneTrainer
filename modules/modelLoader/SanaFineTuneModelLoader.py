from modules.model.SanaModel import SanaModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.sana.SanaEmbeddingLoader import SanaEmbeddingLoader
from modules.modelLoader.sana.SanaModelLoader import SanaModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.ModuleFilter import ModuleFilter


class SanaFineTuneModelLoader(
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
            case ModelType.SANA:
                return "resources/sd_model_spec/sana.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quant_filters: list[ModuleFilter] | None = None,
    ) -> SanaModel | None:
        base_model_loader = SanaModelLoader()
        embedding_loader = SanaEmbeddingLoader()

        model = SanaModel(model_type=model_type)

        self._load_internal_data(model, model_names.base_model)
        model.model_spec = self._load_default_model_spec(model_type)

        base_model_loader.load(model, model_type, model_names, weight_dtypes, quant_filters)
        embedding_loader.load(model, model_names.base_model, model_names)

        return model
