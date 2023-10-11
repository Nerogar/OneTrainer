from abc import ABCMeta
from typing import Any

from torch import Tensor

from modules.module.LoRAModule import LoRAModuleWrapper


class ModelLoaderLoRAMixin(metaclass=ABCMeta):

    def _get_lora_rank(
            self,
            state_dict: dict,
    ) -> int:
        for name, state in state_dict.items():
            if name.endswith("lora_down.weight"):
                return state.shape[0]

    def _load_lora_with_prefix(
            self,
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
