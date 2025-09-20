from modules.model.QwenModel import QwenModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.modelLoader.qwen.QwenModelLoader import QwenModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class QwenFineTuneModelLoader(
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
            case ModelType.QWEN:
                return "resources/sd_model_spec/qwen.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> QwenModel | None:
        base_model_loader = QwenModelLoader()
        model = QwenModel(model_type=model_type)

        self._load_internal_data(model, model_names.base_model)
        model.model_spec = self._load_default_model_spec(model_type)

        base_model_loader.load(model, model_type, model_names, weight_dtypes)

        return model
