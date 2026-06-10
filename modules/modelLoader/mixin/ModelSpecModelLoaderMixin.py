import contextlib
import json
from abc import ABCMeta

from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec

from safetensors import safe_open


class ModelSpecModelLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        return None

    def _load_default_model_spec(
            self,
            model_type: ModelType,
            safetensors_file_name: str | None = None,
    ) -> ModelSpec:
        model_spec = None

        model_spec_name = self._default_model_spec_name(model_type)
        if model_spec_name:
            with open(model_spec_name, "r", encoding="utf-8") as model_spec_file:
                model_spec = ModelSpec.from_dict(json.load(model_spec_file))
        else:
            model_spec = ModelSpec()

        if safetensors_file_name:
            with contextlib.suppress(Exception), safe_open(safetensors_file_name, framework="pt") as f:
                if "modelspec.sai_model_spec" in f.metadata():
                    model_spec = ModelSpec.from_dict(f.metadata())

        return model_spec
