from collections.abc import Iterator

import torch
from torch import Tensor


def combine(left: str, right: str) -> str:
    if left == "":
        return right
    elif right == "":
        return left
    else:
        return left + "." + right


def map_prefix_range(
        in_states: dict[str, Tensor],
        map_back: bool,
        out_prefix: str,
        in_prefix: str,
) -> Iterator[int]:
    prefix = out_prefix if map_back else in_prefix

    i = 0
    while any(key.startswith(f"{prefix}.{i}") for key in in_states):
        yield i
        i += 1


def map_prefix(
        in_states: dict[str, Tensor],
        map_back: bool,
        out_prefix: str,
        in_prefix: str,
        swap_chunks: bool = False,
) -> dict[str, Tensor]:
    out_states = {}

    if map_back:
        in_prefix, out_prefix = out_prefix, in_prefix

    for key in in_states:
        if key.startswith(in_prefix):
            name = key.removeprefix(in_prefix)
            tensor = in_states[key]
            if swap_chunks and name == '.lora_up.weight':
                chunk_0, chunk_1 = tensor.chunk(2, dim=0)
                tensor = torch.cat([chunk_1, chunk_0], dim=0)
            out_key = out_prefix + name
            out_states[out_key] = tensor

    return out_states
