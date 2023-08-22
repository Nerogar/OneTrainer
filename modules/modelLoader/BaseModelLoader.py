from abc import ABCMeta, abstractmethod
from typing import Any

from torch import Tensor

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class BaseModelLoader(metaclass=ABCMeta):
    @staticmethod
    def _create_default_model_spec(
            model_type: ModelType,
    ) -> ModelSpec:
        return ModelSpec()

    @staticmethod
    def _get_lora_rank(state_dict: dict) -> int:
        for name, state in state_dict.items():
            if name.endswith("lora_down.weight"):
                return state.shape[0]

    @staticmethod
    def _load_lora_with_prefix(
            module: Any,
            state_dict: dict[str, Tensor],
            prefix: str,
            rank: int,
            module_filter: list[str] = None,
    ):
        if any(key.startswith(prefix) for key in state_dict.keys()):
            lora = LoRAModuleWrapper(
                orig_module=module,
                rank=rank,
                prefix=prefix,
                module_filter=module_filter,
            )
            lora.load_state_dict(state_dict)
            return lora
        else:
            return None

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> BaseModel | None:
        pass
