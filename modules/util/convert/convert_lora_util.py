import torch
from torch import Tensor


class LoraConversionKeySet:
    def __init__(
            self,
            omi_prefix: str,
            diffusers_prefix: str,
            swap_chunks: bool = False,
    ):
        self.omi_prefix = omi_prefix
        self.legacy_diffusers_prefix = diffusers_prefix.replace('.', '_')
        self.diffusers_prefix = diffusers_prefix

        self.swap_chunks = swap_chunks


def combine(left: str, right: str) -> str:
    if left == "":
        return right
    elif right == "":
        return left
    else:
        return left + "." + right


def map_prefix_range(
        omi_prefix: str,
        diffusers_prefix: str,
) -> list[LoraConversionKeySet]:
    # 100 should be a safe upper bound. increase if it's not enough in the future
    return [LoraConversionKeySet(f"{omi_prefix}.{i}", f"{diffusers_prefix}.{i}") for i in range(100)]


def __convert(
        state_dict: dict[str, Tensor],
        key_sets: list[LoraConversionKeySet],
        target: str,
) -> dict[str, Tensor]:
    out_states = {}

    # TODO: maybe replace with a non O(n^2) algorithm
    for key_set in key_sets:
        for key, tensor in state_dict.items():
            in_prefixes = []
            out_prefix = ''

            if target == 'omi':
                in_prefixes = [key_set.diffusers_prefix, key_set.legacy_diffusers_prefix]
                out_prefix = key_set.omi_prefix
            elif target == 'diffusers':
                in_prefixes = [key_set.omi_prefix]
                out_prefix = key_set.diffusers_prefix
            elif target == 'legacy_diffusers':
                in_prefixes = [key_set.diffusers_prefix]
                out_prefix = key_set.legacy_diffusers_prefix

            if any(key.startswith(p) for p in in_prefixes):
                name = key
                for p in in_prefixes:
                    name = name.removeprefix(p)
                if key_set.swap_chunks and name == '.lora_up.weight':
                    chunk_0, chunk_1 = tensor.chunk(2, dim=0)
                    tensor = torch.cat([chunk_1, chunk_0], dim=0)

                out_states[out_prefix + name] = tensor

    return out_states


def convert_to_omi(
        state_dict: dict[str, Tensor],
        key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    return __convert(state_dict, key_sets, 'omi')

def convert_to_diffusers(
        state_dict: dict[str, Tensor],
        key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    return __convert(state_dict, key_sets, 'diffusers')

def convert_to_legacy_diffusers(
        state_dict: dict[str, Tensor],
        key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    return __convert(state_dict, key_sets, 'legacy_diffusers')
